"""Train an Equinox policy/value model using exact-game mctx self-play."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from connect4_jax.checkpoint import DEFAULT_MODEL_PATH, load_checkpoint, restore_training, save_checkpoint
from connect4_jax.config import Config
from connect4_jax.network import PolicyValueNet
from connect4_jax.replay import Replay
from connect4_jax.search import SearchConfig, tree_memory_bytes
from connect4_jax.training import Runner, make_optimizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu", "tpu"), default="auto")
    parser.add_argument("--cpu", action="store_true", help="Alias for --platform cpu")
    parser.add_argument("--devices", type=int, help="Number of local devices; default: all on selected platform")
    parser.add_argument("--iterations", type=int, default=1000, help="Additional iterations when resuming")
    parser.add_argument("--games-per-device", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--search-policy", choices=("puct", "gumbel"), default="puct")
    parser.add_argument("--opening-fraction", type=float, default=0.0, help="Fraction of self-play games using random legal prefixes")
    parser.add_argument("--opening-plies", type=int, default=8, help="Maximum random prefix length (1..N)")
    parser.add_argument("--preset", choices=("improve", "larger"), help="Recommended experiments; explicit options override preset values")
    parser.add_argument("--search-memory-mb", type=float, default=512, help="Persistent tree budget per device; excludes activations/temporaries")
    parser.add_argument("--batch-size", type=int, default=None, help="Global learner batch")
    parser.add_argument("--updates-per-iteration", type=int, default=20)
    parser.add_argument("--warmup-positions", type=int, default=4096)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--temperature-moves", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--connect", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument("--init-from", help="Start a new experiment with weights from an Equinox checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--export", help="Optional lightweight inference checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", help="Retain full resumable iteration snapshots here")
    parser.add_argument("--keep-checkpoints", type=int, default=10, help="Number of snapshots to retain; 0 keeps all")
    parser.add_argument("--eval-every", type=int, default=0, help="Fixed-opponent evaluation interval; 0 disables")
    parser.add_argument("--eval-games", type=int, default=10, help="Games per seat per opponent, for each of raw/search modes")
    parser.add_argument("--eval-depths", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--eval-opening-plies", type=int, default=4)
    parser.add_argument("--eval-simulations", type=int, default=128)
    parser.add_argument("--best-output", help="Best inference checkpoint; default: output stem + _best.eqx")
    parser.add_argument("--log", help="Append-only JSONL metrics")
    args = parser.parse_args(argv)
    # ArgumentParser records explicit options robustly, including --option=value.
    import sys
    supplied = sys.argv[1:] if argv is None else argv
    args.explicit_options = {arg.split("=", 1)[0][2:].replace("-", "_")
                             for arg in supplied if arg.startswith("--")}
    if args.preset:
        recommended = dict(search_policy="gumbel", simulations=128, opening_fraction=0.2,
                           learning_rate=0.0003, checkpoint_every=100, eval_every=100)
        if args.preset == "larger":
            recommended.update(channels=64, blocks=2, learning_rate=0.001)
        for name, value in recommended.items():
            if name not in args.explicit_options:
                setattr(args, name, value)
                args.explicit_options.add(name)
    return args


def train(args):
    restored = None
    if args.resume and args.init_from:
        raise ValueError("Use either resume or init-from")
    if args.resume:
        model, restored = load_checkpoint(args.resume)
        saved = restored.get("training_args", {})
        for name in ("games_per_device", "simulations", "batch_size", "updates_per_iteration",
                     "warmup_positions", "replay_capacity", "learning_rate", "weight_decay",
                     "temperature_moves", "c_puct", "dirichlet_alpha", "noise_fraction", "precision", "seed",
                     "search_memory_mb", "search_policy", "opening_fraction", "opening_plies",
                     "eval_every", "eval_games", "eval_depths", "eval_seed", "eval_opening_plies", "eval_simulations",
                     "checkpoint_every", "keep_checkpoints"):
            if name not in args.explicit_options and name in saved:
                setattr(args, name, saved[name])
        if "output" not in args.explicit_options:
            for name in ("checkpoint_dir", "best_output"):
                if name not in args.explicit_options and saved.get(name):
                    setattr(args, name, saved[name])
        config = model.config
    elif args.init_from:
        model, _ = load_checkpoint(args.init_from)
        config = model.config
    else:
        config = Config(args.rows, args.cols, args.connect, args.channels, args.blocks)
        model = None
    args.output = args.output or args.resume or DEFAULT_MODEL_PATH
    if args.preset and not args.checkpoint_dir:
        args.checkpoint_dir = str(Path(args.output).parent / (Path(args.output).stem + "_checkpoints"))
    if args.resume or args.init_from:
        for name in ("rows", "cols", "connect", "channels", "blocks"):
            if name in args.explicit_options and getattr(args, name) != getattr(config, name):
                raise ValueError(f"{name} cannot change when loading existing weights")
    platform = "cpu" if args.cpu else None if args.platform == "auto" else args.platform
    devices = jax.local_devices(backend=platform)
    if args.devices is not None:
        if not 1 <= args.devices <= len(devices):
            raise ValueError(f"Requested {args.devices} devices; found {len(devices)}")
        devices = devices[:args.devices]
    if jax.process_count() != 1:
        raise ValueError("This runner supports a single host; multi-host training is not implemented")
    accelerator = devices[0].platform
    if args.games_per_device is None:
        args.games_per_device = 64 if accelerator == "tpu" else 32 if accelerator == "gpu" else 2
    if args.batch_size is None:
        args.batch_size = 1024 if accelerator == "tpu" else 512 if accelerator == "gpu" else 32
    for name in ("iterations", "games_per_device", "simulations", "batch_size", "updates_per_iteration",
                 "replay_capacity", "checkpoint_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warmup_positions < 0 or args.temperature_moves < 0 or args.seed < 0:
        raise ValueError("Warmup, temperature moves, and seed must be nonnegative")
    if args.batch_size % len(devices):
        raise ValueError("Global batch size must be divisible by number of devices")
    if args.replay_capacity < max(args.batch_size, args.warmup_positions):
        raise ValueError("Replay capacity must accommodate batch size and warmup")
    if args.export and Path(args.export).resolve() == Path(args.output).resolve():
        raise ValueError("Export and resumable checkpoint must use different paths")
    if args.eval_every < 0 or args.eval_games < 1 or args.eval_simulations < 1 or args.keep_checkpoints < 0:
        raise ValueError("Evaluation/retention settings must be nonnegative with positive games/simulations")
    if any(depth < 1 for depth in args.eval_depths) or not 0 <= args.eval_opening_plies <= 12:
        raise ValueError("Evaluation depths must be positive and opening plies in [0,12]")
    if args.eval_every and (config.rows, config.cols, config.connect) != (6, 7, 4):
        raise ValueError("Disable --eval-every for nonstandard boards")
    if args.search_policy == "gumbel" and (args.simulations < config.cols or (args.eval_every and args.eval_simulations < config.cols)):
        raise ValueError("Gumbel simulations must be at least the number of columns")
    args.best_output = args.best_output or str(Path(args.output).with_name(Path(args.output).stem + "_best.eqx"))
    if args.eval_every and any(Path(args.best_output).resolve() == Path(p).resolve()
                              for p in (args.output, args.export) if p):
        raise ValueError("Best checkpoint must have a separate path from latest/export")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    if args.precision == "bf16" and accelerator != "tpu":
        raise ValueError("BF16 training is enabled for TPU only; use fp32 on T4/CPU")
    dtype = jnp.bfloat16 if args.precision == "bf16" else jnp.float32
    key = jax.random.PRNGKey(args.seed)
    rng = np.random.default_rng(args.seed)
    if model is None:
        key, init_key = jax.random.split(key)
        with jax.default_device(devices[0]):
            model = PolicyValueNet(config, init_key)
    settings = SearchConfig(args.simulations, args.c_puct, args.dirichlet_alpha, args.noise_fraction, args.search_policy)
    search_mib = tree_memory_bytes(config, args.games_per_device, args.simulations, args.search_policy) / 2**20
    if not np.isfinite(args.search_memory_mb) or args.search_memory_mb <= 0:
        raise ValueError("Search memory budget must be positive and finite")
    if search_mib > args.search_memory_mb:
        raise ValueError(f"Persistent trees require {search_mib:.1f} MiB per device; budget is {args.search_memory_mb:.1f} MiB")
    optimizer = make_optimizer(args.learning_rate, args.weight_decay)
    runner = Runner(model, optimizer, devices, settings, args.games_per_device, args.temperature_moves, dtype,
                    args.opening_fraction, args.opening_plies)
    replay = Replay(args.replay_capacity, config)
    iteration = games_played = updates_done = 0
    if restored:
        if restored.get("device_count") != len(devices) or restored.get("platform") != accelerator:
            raise ValueError("Resume requires the saved device count/platform; use --init-from for a new experiment")
        if args.replay_capacity != restored["training_args"]["replay_capacity"]:
            raise ValueError("Resume requires the saved replay capacity")
        runner.restore_optimizer(restore_training(args.resume, runner.optimizer_state, replay))
        rng.bit_generator.state = restored["numpy_rng"]
        key = jnp.asarray(restored["jax_key"], jnp.uint32)
        iteration, games_played = restored["iteration"], restored["games_played"]
        updates_done = restored["updates_done"]
    print(json.dumps(dict(event="startup", platform=accelerator, devices=len(devices),
                          games_per_iteration=runner.games, batch_size=args.batch_size,
                          precision=args.precision, config=asdict(config),
                          search_policy=settings.policy, opening_fraction=args.opening_fraction,
                          search_memory_mib_per_device=search_mib)), flush=True)
    saved_args = {k: v for k, v in vars(args).items() if k != "explicit_options"}
    saved_args.update(asdict(config))
    evaluation_signature = dict(games=args.eval_games, depths=args.eval_depths, seed=args.eval_seed,
                                opening_plies=args.eval_opening_plies, simulations=args.eval_simulations,
                                search_policy=args.search_policy, c_puct=args.c_puct)
    best_evaluation = (restored.get("best_evaluation") if restored and
                       restored.get("evaluation_signature") == evaluation_signature and
                       Path(restored.get("training_args", {}).get("best_output") or "").resolve() == Path(args.best_output).resolve() and
                       Path(args.best_output).exists() else None)
    def checkpoint():
        snapshot = None
        if args.checkpoint_dir:
            directory = Path(args.checkpoint_dir)
            prefix = Path(args.output).stem + ".iter-"
            snapshot = directory / f"{prefix}{iteration:07d}.eqx"
            if snapshot.exists():
                raise ValueError(f"Snapshot already exists; choose a new experiment directory: {snapshot}")
        metadata = dict(iteration=iteration, games_played=games_played, updates_done=updates_done,
                        training_args=saved_args, device_count=len(devices), platform=accelerator,
                        numpy_rng=rng.bit_generator.state, jax_key=np.asarray(key).tolist(), search=asdict(settings),
                        evaluation_signature=evaluation_signature, best_evaluation=best_evaluation)
        save_checkpoint(args.output, runner.model, metadata=metadata,
                        optimizer_state=runner.optimizer_state, replay=replay)
        if args.export:
            if Path(args.export).resolve() == Path(args.output).resolve():
                raise ValueError("Export and resumable checkpoint must use different paths")
            save_checkpoint(args.export, runner.model, metadata=dict(iteration=iteration, search=asdict(settings)))
        if snapshot is not None:
            directory.mkdir(parents=True, exist_ok=True)
            save_checkpoint(snapshot, runner.model, metadata=metadata,
                            optimizer_state=runner.optimizer_state, replay=replay)
            snapshots = sorted(p for p in directory.iterdir() if p.is_file() and p.name.startswith(prefix)
                               and p.suffix == ".eqx" and p.name[len(prefix):-4].isdigit())
            if args.keep_checkpoints:
                for expired in snapshots[:-args.keep_checkpoints]:
                    expired.unlink()

    start_iteration = iteration
    start = time.perf_counter()
    learner_warmed = False
    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        collect_start = time.perf_counter()
        key, collect_key = jax.random.split(key)
        trajectory = jax.device_get(runner.collect(collect_key))  # synchronises timed collection
        replay.add(trajectory)
        collection_seconds = time.perf_counter() - collect_start
        games_played += runner.games
        positions = int(trajectory.valid.sum())
        train_start = time.perf_counter()
        losses = []
        if len(replay) >= max(args.batch_size, args.warmup_positions):
            for _ in range(args.updates_per_iteration):
                loss = np.asarray(runner.update(replay.sample(args.batch_size, rng)))
                if not np.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite training loss; checkpoint not replaced")
                losses.append(loss)
                updates_done += 1
        training_seconds = time.perf_counter() - train_start
        current_evaluation = None
        evaluation_seconds = 0.0
        if args.eval_every and (iteration % args.eval_every == 0 or iteration == start_iteration + args.iterations):
            from connect4_jax.evaluation import evaluate
            eval_start = time.perf_counter()
            current_evaluation = evaluate(runner.model, **evaluation_signature)
            current_evaluation["iteration"] = iteration
            rank = (current_evaluation["selection_score"], current_evaluation["raw_selection_score"])
            previous = (best_evaluation["selection_score"], best_evaluation["raw_selection_score"]) if best_evaluation else (-1, -1)
            if rank > previous:
                best_evaluation = current_evaluation
                save_checkpoint(args.best_output, runner.model,
                                metadata=dict(iteration=iteration, search=asdict(settings), evaluation=current_evaluation))
            report_dir = Path(args.output).parent / (Path(args.output).stem + "_evaluations")
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"iteration_{iteration:07d}.json").write_text(json.dumps(current_evaluation, indent=2), encoding="utf-8")
            evaluation_seconds = time.perf_counter() - eval_start
        training_compilation = bool(losses) and not learner_warmed
        learner_warmed |= bool(losses)
        record = dict(iteration=iteration, games_played=games_played, updates_done=updates_done,
                      config=asdict(config), learning_rate=args.learning_rate, simulations=args.simulations,
                      opening_fraction=args.opening_fraction,
                      positions=positions, replay_positions=len(replay),
                      opening_games=int((trajectory.opening_plies > 0).sum()),
                      mean_opening_plies=float(trajectory.opening_plies.mean()), search_policy=settings.policy,
                      self_play=dict(first_wins=int((trajectory.winners == 1).sum()),
                                     second_wins=int((trajectory.winners == 2).sum()),
                                     draws=int((trajectory.winners == 0).sum())),
                      collection_seconds=collection_seconds, training_seconds=training_seconds,
                      positions_per_second=positions / max(collection_seconds, 1e-9),
                      games_per_minute=runner.games * 60 / max(collection_seconds, 1e-9),
                      loss=np.mean(losses, axis=0).tolist() if losses else None,
                      collection_includes_compilation=iteration == start_iteration + 1,
                      training_includes_compilation=training_compilation,
                      evaluation_seconds=evaluation_seconds, evaluation=current_evaluation,
                      best_iteration=best_evaluation["iteration"] if best_evaluation else None,
                      elapsed_seconds=time.perf_counter() - start)
        line = json.dumps(record)
        print(line, flush=True)
        if args.log:
            path = Path(args.log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        if iteration % args.checkpoint_every == 0 or iteration == start_iteration + args.iterations:
            checkpoint()
    return runner.model


if __name__ == "__main__":
    train(parse_args())
