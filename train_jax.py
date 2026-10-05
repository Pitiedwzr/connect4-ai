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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu", "tpu"), default="auto")
    parser.add_argument("--cpu", action="store_true", help="Alias for --platform cpu")
    parser.add_argument("--devices", type=int, help="Number of local devices; default: all on selected platform")
    parser.add_argument("--iterations", type=int, default=1000, help="Additional iterations when resuming")
    parser.add_argument("--games-per-device", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=64)
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
    parser.add_argument("--log", help="Append-only JSONL metrics")
    args = parser.parse_args(argv)
    # ArgumentParser records explicit options robustly, including --option=value.
    import sys
    supplied = sys.argv[1:] if argv is None else argv
    args.explicit_options = {arg.split("=", 1)[0][2:].replace("-", "_")
                             for arg in supplied if arg.startswith("--")}
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
                     "search_memory_mb"):
            if name not in args.explicit_options and name in saved:
                setattr(args, name, saved[name])
        config = model.config
    elif args.init_from:
        model, _ = load_checkpoint(args.init_from)
        config = model.config
    else:
        config = Config(args.rows, args.cols, args.connect, args.channels, args.blocks)
        model = None
    args.output = args.output or args.resume or DEFAULT_MODEL_PATH
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
    settings = SearchConfig(args.simulations, args.c_puct, args.dirichlet_alpha, args.noise_fraction)
    search_mib = tree_memory_bytes(config, args.games_per_device, args.simulations) / 2**20
    if not np.isfinite(args.search_memory_mb) or args.search_memory_mb <= 0:
        raise ValueError("Search memory budget must be positive and finite")
    if search_mib > args.search_memory_mb:
        raise ValueError(f"Persistent trees require {search_mib:.1f} MiB per device; budget is {args.search_memory_mb:.1f} MiB")
    optimizer = make_optimizer(args.learning_rate, args.weight_decay)
    runner = Runner(model, optimizer, devices, settings, args.games_per_device, args.temperature_moves, dtype)
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
                          search_memory_mib_per_device=search_mib)), flush=True)
    saved_args = {k: v for k, v in vars(args).items() if k != "explicit_options"}
    saved_args.update(asdict(config))

    def checkpoint():
        metadata = dict(iteration=iteration, games_played=games_played, updates_done=updates_done,
                        training_args=saved_args, device_count=len(devices), platform=accelerator,
                        numpy_rng=rng.bit_generator.state, jax_key=np.asarray(key).tolist())
        save_checkpoint(args.output, runner.model, metadata=metadata,
                        optimizer_state=runner.optimizer_state, replay=replay)
        if args.export:
            if Path(args.export).resolve() == Path(args.output).resolve():
                raise ValueError("Export and resumable checkpoint must use different paths")
            save_checkpoint(args.export, runner.model, metadata=dict(iteration=iteration))

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
        training_compilation = bool(losses) and not learner_warmed
        learner_warmed |= bool(losses)
        record = dict(iteration=iteration, games_played=games_played, updates_done=updates_done,
                      positions=positions, replay_positions=len(replay),
                      self_play=dict(first_wins=int((trajectory.winners == 1).sum()),
                                     second_wins=int((trajectory.winners == 2).sum()),
                                     draws=int((trajectory.winners == 0).sum())),
                      collection_seconds=collection_seconds, training_seconds=training_seconds,
                      positions_per_second=positions / max(collection_seconds, 1e-9),
                      games_per_minute=runner.games * 60 / max(collection_seconds, 1e-9),
                      loss=np.mean(losses, axis=0).tolist() if losses else None,
                      collection_includes_compilation=iteration == start_iteration + 1,
                      training_includes_compilation=training_compilation,
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
