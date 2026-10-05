"""Train Double DQN on CPU or Accelerate-managed GPUs."""
import argparse
from collections import deque
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from agent import DQNAgent, ReplayMemory
from benchmark import evaluate_network, network_move, teacher_move
from game import (create_board, drop_piece, get_candidate_moves, get_next_open_row,
                  get_state_tensor, get_valid_locations, winning_move)

AI_PIECE, OPPONENT_PIECE = 1, 2


def get_opponent_action(board, valid_moves, opp_type, opponent_net=None, device="cpu", depths=(2,)):
    candidates = get_candidate_moves(board, OPPONENT_PIECE, AI_PIECE)
    if len(candidates) == 1:
        return candidates[0]
    draw = random.random()
    if opp_type == "SelfPlay" and opponent_net is not None and draw < 0.60:
        return network_move(opponent_net, board, OPPONENT_PIECE, AI_PIECE)
    if opp_type == "Minimax" or (opp_type == "SelfPlay" and draw < 0.90):
        return teacher_move(board, OPPONENT_PIECE, AI_PIECE, random.choice(depths))
    return random.choice(candidates)


def pretrain_from_search(agent, positions, batch_size, depth):
    """Optional action pretraining from search labels, not invented Q rewards.

    Legal random positions cover both seats. Cross entropy teaches preferred
    actions before sparse-reward learning. DDP ranks perform equal batch counts.
    """
    losses = []
    for offset in range(0, positions, batch_size):
        states, labels, legal_masks = [], [], []
        for _ in range(min(batch_size, positions - offset)):
            while True:
                board, piece = create_board(), 1
                terminal = False
                for _ in range(random.randrange(25)):
                    col = random.choice(get_valid_locations(board))
                    drop_piece(board, get_next_open_row(board, col), col, piece)
                    if winning_move(board, piece):
                        terminal = True
                        break
                    piece = 3 - piece
                if not terminal:
                    break
            state = get_state_tensor(board, piece, 3 - piece)
            label = teacher_move(board, piece, 3 - piece, depth)
            mask = torch.zeros(7, dtype=torch.bool)
            mask[get_valid_locations(board)] = True
            states.extend((state, torch.flip(state, dims=[3])))
            labels.extend((label, 6 - label))
            legal_masks.extend((mask, torch.flip(mask, dims=[0])))
        agent.policy_net.train()
        logits = agent.policy_net(torch.cat(states).to(agent.device))
        logits = logits.masked_fill(~torch.stack(legal_masks).to(agent.device), -torch.inf)
        loss = F.cross_entropy(logits, torch.tensor(labels, device=agent.device))
        agent.optimizer.zero_grad()
        agent.accelerator.backward(loss)
        agent.accelerator.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
        agent.optimizer.step()
        losses.append(loss.item())
    agent.update_target_network(tau=1.0)
    return sum(losses) / len(losses) if losses else 0.0


def train(args=None):
    if args is None:
        args = parse_args()
    accelerator = Accelerator(cpu=args.cpu)
    if accelerator.device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    seed = args.seed + accelerator.process_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DQNAgent(accelerator=accelerator)
    agent.policy_net, agent.optimizer = accelerator.prepare(agent.policy_net, agent.optimizer)
    # DDP broadcasts policy parameters on prepare; synchronize each local target.
    agent.update_target_network(tau=1.0)
    if args.pretrain_positions:
        loss = pretrain_from_search(agent, args.pretrain_positions, args.batch_size, args.pretrain_depth)
        accelerator.print(f"Search pretraining: {args.pretrain_positions} positions per process, loss={loss:.4f}")

    memory = ReplayMemory(capacity=50000)
    pool = deque(maxlen=args.opponent_pool_size)

    def snapshot():
        net = copy.deepcopy(agent._get_unwrapped_policy())
        net.eval()
        net.requires_grad_(False)
        pool.append(net)

    snapshot()
    processes = accelerator.num_processes
    # Identical loop lengths are necessary for distributed collective operations.
    rounds = math.ceil(args.episodes / processes)
    log_every = max(1, math.ceil(100 / processes))
    snapshot_every = max(1, math.ceil(args.snapshot_every / processes))
    eval_every = max(1, math.ceil(args.eval_every / processes)) if args.eval_every else 0
    counts = np.zeros(4, dtype=np.float32)  # wins, losses, draws, summed loss
    update_count = 0
    accelerator.print(f"Training {rounds * processes} games on {processes} process(es), device={accelerator.device}")

    for episode in range(1, rounds + 1):
        progress = (episode - 1) / max(1, rounds - 1)
        max_depth = min(args.minimax_max_depth, 2 + int(progress * 3))
        depths = tuple(range(2, max_depth + 1))
        # Decay by global games rather than local games for multi-GPU consistency.
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay ** ((episode - 1) * processes))
        teacher_probability = args.teacher_probability * (1.0 - progress)
        opponent_net = random.choice(list(pool))
        board, turn, state, action, steps = create_board(), random.choice([0, 1]), None, None, 0
        done = False
        while not done:
            legal = get_valid_locations(board)
            if turn == 0:
                state = get_state_tensor(board, AI_PIECE, OPPONENT_PIECE)
                if random.random() < teacher_probability:
                    action = teacher_move(board, AI_PIECE, OPPONENT_PIECE, max_depth)
                else:
                    action = agent.act(state, legal, board, AI_PIECE, OPPONENT_PIECE)
                drop_piece(board, get_next_open_row(board, action), action, AI_PIECE)
                steps += 1
                if winning_move(board, AI_PIECE):
                    counts[0] += 1
                    done = True
                    reward = 1.0
                elif not get_valid_locations(board):
                    counts[2] += 1
                    done = True
                    reward = 0.0
                if done:
                    memory.push_with_symmetry(state, action, reward,
                                              get_state_tensor(board, AI_PIECE, OPPONENT_PIECE), True)
            else:
                col = get_opponent_action(board, legal, args.opponent, opponent_net,
                                          agent.device, depths)
                drop_piece(board, get_next_open_row(board, col), col, OPPONENT_PIECE)
                reward = 0.0
                if winning_move(board, OPPONENT_PIECE):
                    counts[1] += 1
                    done, reward = True, -1.0
                elif not get_valid_locations(board):
                    counts[2] += 1
                    done = True
                if state is not None:
                    memory.push_with_symmetry(state, action, reward,
                                              get_state_tensor(board, AI_PIECE, OPPONENT_PIECE), done)
            turn = 1 - turn

        # Accelerate supports sum/mean, not min; every rank must be ready.
        ready = torch.tensor([float(len(memory) >= max(args.warmup_steps, args.batch_size))], device=agent.device)
        all_ready = accelerator.reduce(ready, reduction="sum").item() == processes
        steps_tensor = torch.tensor([float(steps)], device=agent.device)
        avg_steps = accelerator.reduce(steps_tensor, reduction="mean").item()
        if all_ready:
            for _ in range(max(1, round(avg_steps / args.learn_frequency))):
                counts[3] += agent.learn(memory, args.batch_size)
                update_count += 1
                agent.update_target_network(tau=0.005)

        if episode % snapshot_every == 0:
            snapshot()
        if episode % log_every == 0 or episode == rounds:
            metrics = accelerator.reduce(torch.tensor(counts, device=agent.device), reduction="sum").cpu().tolist()
            updates = accelerator.reduce(torch.tensor([float(update_count)], device=agent.device), reduction="sum").item()
            games = sum(metrics[:3])
            accelerator.print(f"Games={episode * processes} | exploratory W/L/D={metrics[:3]} | "
                              f"win rate={metrics[0] / max(1, games):.2%} | "
                              f"loss={metrics[3] / max(1, updates):.4f} | epsilon={agent.epsilon:.3f}")
            counts[:] = 0
            update_count = 0
        if eval_every and (episode % eval_every == 0 or episode == rounds):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                report = evaluate_network(agent._get_unwrapped_policy(), args.eval_games, args.seed + 10000)
                print("Frozen evaluation:", json.dumps(report))
            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent._get_unwrapped_policy().state_dict(), output)
        print(f"Saved model to {output}")
    return agent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--learn-frequency", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="model/connect4_model_selfplay.pth")
    parser.add_argument("--opponent", choices=["SelfPlay", "Minimax", "Random"], default="SelfPlay")
    parser.add_argument("--minimax-max-depth", type=int, default=4)
    parser.add_argument("--teacher-probability", type=float, default=0.25)
    parser.add_argument("--pretrain-positions", type=int, default=0)
    parser.add_argument("--pretrain-depth", type=int, default=3)
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument("--snapshot-every", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=1000, help="Global games between frozen evaluations; 0 disables")
    parser.add_argument("--eval-games", type=int, default=20)
    args = parser.parse_args()
    for name in ("episodes", "batch_size", "learn_frequency", "cpu_threads", "pretrain_depth",
                 "opponent_pool_size", "snapshot_every", "eval_games"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0 or args.pretrain_positions < 0 or args.eval_every < 0:
        parser.error("warmup steps, pretrain positions and eval interval must be nonnegative")
    if args.minimax_max_depth < 2 or not 0 <= args.teacher_probability <= 1:
        parser.error("minimax depth must be >=2 and teacher probability in [0,1]")
    return args


if __name__ == "__main__":
    train()
