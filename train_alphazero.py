"""Train an AlphaZero-style agent using batched, search-guided self-play."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time
import sys

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F

from alphazero import (AlphaZeroConfig, AlphaZeroNet, DEFAULT_MODEL_PATH, MCTS,
                       Position, load_checkpoint, save_checkpoint)


class SelfPlayReplay:
    def __init__(self, capacity=50000):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.examples = deque(maxlen=capacity)

    def add(self, examples, mirror=True):
        for state, policy, outcome in examples:
            state = torch.as_tensor(state, dtype=torch.float32).clone()
            policy = torch.as_tensor(policy, dtype=torch.float32).clone()
            self.examples.append((state, policy, float(outcome)))
            if mirror:
                self.examples.append((torch.flip(state, [2]), torch.flip(policy, [0]), float(outcome)))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.examples), size=batch_size, replace=False)
        states, policies, outcomes = zip(*(self.examples[int(index)] for index in indices))
        return torch.stack(states), torch.stack(policies), torch.tensor(outcomes, dtype=torch.float32)

    def state_dict(self):
        if not self.examples:
            return None
        states, policies, outcomes = zip(*self.examples)
        return dict(states=torch.stack(states), policies=torch.stack(policies),
                    outcomes=torch.tensor(outcomes, dtype=torch.float32))

    def load_state_dict(self, state):
        self.examples.clear()
        if state is not None:
            self.examples.extend(zip(state["states"].cpu(), state["policies"].cpu(),
                                     state["outcomes"].cpu().tolist()))

    def __len__(self):
        return len(self.examples)


def self_play_games(model, games=8, simulations=64, temperature_moves=10,
                    c_puct=1.5, dirichlet_alpha=0.3, noise_fraction=0.25, rng=None):
    """Batch one leaf per active game at each search simulation.

    Model weights stay frozen for the entire collection. Both seats contribute
    (position, root visit distribution, undiscounted final outcome) examples.
    """
    if games < 1 or temperature_moves < 0:
        raise ValueError("games must be positive and temperature_moves nonnegative")
    rng = rng or np.random.default_rng()
    model.eval()
    positions = [Position.empty(model.config) for _ in range(games)]
    searches = [MCTS(model, simulations, c_puct,
                     rng=np.random.default_rng(int(rng.integers(2**32))),
                     dirichlet_alpha=dirichlet_alpha, noise_fraction=noise_fraction)
                for _ in range(games)]
    histories = [[] for _ in range(games)]
    active = list(range(games))
    while active:
        results = MCTS.search_batch([searches[index] for index in active],
                                   [positions[index] for index in active], add_noise=True)
        continuing = []
        for index, result in zip(active, results):
            position = positions[index]
            histories[index].append((position.encode(), result.policy.copy(), position.to_play))
            temperature = 1.0 if position.ply < temperature_moves else 0.0
            action = result.action(searches[index].rng, temperature)
            positions[index] = position.play(action)
            searches[index].advance(action)
            if positions[index].terminal_value() is None:
                continuing.append(index)
        active = continuing
    examples = []
    for position, history in zip(positions, histories):
        for state, policy, side in history:
            outcome = (0.0 if not position.winner else
                       1.0 if position.winner == side else -1.0)
            examples.append((state, policy, outcome))
    return examples, dict(first_wins=sum(p.winner == 1 for p in positions),
                          second_wins=sum(p.winner == 2 for p in positions),
                          draws=sum(p.winner == 0 for p in positions),
                          positions=len(examples))


def policy_value_loss(logits, values, states, target_policies, target_outcomes):
    """Masked cross entropy on search visits plus MSE on final game outcomes."""
    legal = ~states[:, :, -1, :].bool().any(dim=1)
    if not legal.any(dim=1).all():
        raise ValueError("Terminal states do not belong in the training replay")
    if (target_policies.masked_select(~legal) > 0).any():
        raise ValueError("Search targets must assign zero mass to illegal moves")
    log_probs = F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=1)
    # Zero illegal entries before multiplication: 0 * -inf would produce NaN.
    policy_loss = -(target_policies * log_probs.masked_fill(~legal, 0.0)).sum(dim=1).mean()
    value_loss = F.mse_loss(values.reshape(-1), target_outcomes)
    return policy_loss + value_loss, policy_loss, value_loss


def train(args):
    accelerator = Accelerator(cpu=args.cpu)
    if accelerator.device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed + accelerator.process_index)
    rng = np.random.default_rng(args.seed + accelerator.process_index)
    replay = SelfPlayReplay(args.replay_capacity)
    iteration, games_played = 0, 0
    restored = None
    if args.resume:
        model, restored = load_checkpoint(args.resume, accelerator.device)
        explicit = set(getattr(args, "explicit_options", vars(args).keys()))
        if "output" not in explicit:
            args.output = args.resume
        saved_args = restored.get("training_args", {})
        for name in ("games_per_iteration", "simulations", "batch_size", "updates_per_iteration",
                     "warmup_positions", "replay_capacity", "learning_rate", "weight_decay",
                     "temperature_moves", "c_puct", "dirichlet_alpha", "noise_fraction", "seed", "cpu_threads"):
            if name not in explicit and name in saved_args:
                setattr(args, name, saved_args[name])
        if args.replay_capacity < max(args.batch_size, args.warmup_positions):
            raise ValueError("Resumed replay capacity must accommodate batch size and warmup positions")
        if accelerator.device.type == "cpu":
            torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed + accelerator.process_index)
        rng = np.random.default_rng(args.seed + accelerator.process_index)
        replay = SelfPlayReplay(args.replay_capacity)
        config = model.config
        for name in ("rows", "cols", "connect", "channels", "blocks"):
            setattr(args, name, getattr(config, name))
        iteration = restored.get("iteration", 0)
        games_played = restored.get("games_played", 0)
        # Each process owns a different replay/RNG. Do not clone rank zero's
        # saved stream into all ranks; exact replay resume is single-process.
        if accelerator.num_processes == 1:
            replay.load_state_dict(restored.get("replay"))
            if "numpy_rng" in restored:
                rng.bit_generator.state = restored["numpy_rng"]
            if "torch_rng" in restored:
                torch.set_rng_state(restored["torch_rng"].cpu())
            if torch.cuda.is_available() and "cuda_rng" in restored:
                torch.cuda.set_rng_state_all([state.cpu() for state in restored["cuda_rng"]])
    else:
        config = AlphaZeroConfig(args.rows, args.cols, args.connect, args.channels, args.blocks)
        model = AlphaZeroNet(config).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if restored is not None and "optimizer_state" in restored:
        optimizer.load_state_dict(restored["optimizer_state"])
        # A requested rate applies to subsequent training, including resume.
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay
    model, optimizer = accelerator.prepare(model, optimizer)
    net = accelerator.unwrap_model(model)
    accelerator.print(f"AlphaZero self-play on {accelerator.device}, {accelerator.num_processes} process(es), "
                      f"config={asdict(config)}, simulations={args.simulations}")
    if args.resume and accelerator.num_processes > 1:
        accelerator.print("Distributed resume restores weights/optimizer; local replay and RNG streams restart.")
    start_iteration = iteration
    start_time = time.monotonic()

    def checkpoint(current_iteration):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state = dict(optimizer_state=optimizer.state_dict(), iteration=current_iteration,
                         games_played=games_played, training_args=vars(args),
                         numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                         replay=replay.state_dict())
            if torch.cuda.is_available():
                state["cuda_rng"] = torch.cuda.get_rng_state_all()
            save_checkpoint(args.output, net, **state)
        accelerator.wait_for_everyone()

    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        examples, game_metrics = self_play_games(
            net, args.games_per_iteration, args.simulations, args.temperature_moves,
            args.c_puct, args.dirichlet_alpha, args.noise_fraction, rng)
        replay.add(examples)
        games_played += args.games_per_iteration * accelerator.num_processes
        ready = torch.tensor([float(len(replay) >= max(args.batch_size, args.warmup_positions))],
                             device=accelerator.device)
        all_ready = accelerator.reduce(ready, reduction="sum").item() == accelerator.num_processes
        losses = np.zeros(3, dtype=np.float64)
        updates = 0
        if all_ready:
            for _ in range(args.updates_per_iteration):
                states, policies, outcomes = replay.sample(args.batch_size, rng)
                states = states.to(accelerator.device)
                policies = policies.to(accelerator.device)
                outcomes = outcomes.to(accelerator.device)
                model.train()
                logits, values = model(states)
                loss, policy_loss, value_loss = policy_value_loss(logits, values, states, policies, outcomes)
                optimizer.zero_grad()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses += [loss.item(), policy_loss.item(), value_loss.item()]
                updates += 1
        metrics = torch.tensor([game_metrics["first_wins"], game_metrics["second_wins"],
                                game_metrics["draws"], game_metrics["positions"],
                                *losses, updates], device=accelerator.device, dtype=torch.float64)
        metrics = accelerator.reduce(metrics, reduction="sum").cpu().tolist()
        record = dict(iteration=iteration, games_played=games_played,
                      self_play=dict(first_wins=int(metrics[0]), second_wins=int(metrics[1]),
                                     draws=int(metrics[2]), positions=int(metrics[3])),
                      loss=metrics[4] / max(1, metrics[7]),
                      policy_loss=metrics[5] / max(1, metrics[7]),
                      value_loss=metrics[6] / max(1, metrics[7]), updates=int(metrics[7]),
                      replay_positions=len(replay), elapsed_seconds=round(time.monotonic() - start_time, 2))
        accelerator.print(json.dumps(record))
        if args.log and accelerator.is_main_process:
            log = Path(args.log)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if iteration % args.checkpoint_every == 0 or iteration == start_iteration + args.iterations:
            checkpoint(iteration)
    accelerator.print(f"Saved AlphaZero checkpoint to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000, help="Additional iterations, including on resume")
    parser.add_argument("--games-per-iteration", type=int, default=8, help="Concurrent games per process")
    parser.add_argument("--simulations", type=int, default=64, help="New PUCT simulations per move")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-iteration", type=int, default=20)
    parser.add_argument("--warmup-positions", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--temperature-moves", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--connect", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--resume", help="AlphaZero checkpoint; its network/game configuration is reused")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--log", help="Optional append-only JSONL training log")
    args = parser.parse_args()
    args.explicit_options = [option.split("=", 1)[0][2:].replace("-", "_")
                             for option in sys.argv[1:] if option.startswith("--")]
    for name in ("iterations", "games_per_iteration", "simulations", "batch_size", "updates_per_iteration",
                 "replay_capacity", "cpu_threads", "checkpoint_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.temperature_moves < 0 or args.warmup_positions < 0:
        parser.error("temperature moves and warmup positions must be nonnegative")
    if args.replay_capacity < max(args.batch_size, args.warmup_positions):
        parser.error("replay capacity must accommodate batch size and warmup positions")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.c_puct <= 0 or args.dirichlet_alpha <= 0:
        parser.error("learning rate, c-puct and alpha must be positive; weight decay nonnegative")
    if not 0 <= args.noise_fraction <= 1:
        parser.error("noise fraction must be in [0,1]")
    try:
        AlphaZeroConfig(args.rows, args.cols, args.connect, args.channels, args.blocks)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    train(parse_args())
