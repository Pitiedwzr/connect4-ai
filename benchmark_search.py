"""Compare end-to-end self-play throughput, excluding startup/warmup.

No weights are updated or saved. Benchmark on the target CUDA hardware before
choosing an experimental search backend for a long training run.
"""
import argparse
from contextlib import nullcontext
import json
import statistics
import time

import numpy as np
import torch

from alphazero import AlphaZeroConfig, AlphaZeroNet, load_checkpoint
from train_alphazero import self_play_games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=("python", "cpu", "cuda"), default=["python", "cpu"])
    parser.add_argument("--model", help="Optional existing AlphaZero checkpoint")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=("no", "fp16"), default="no")
    parser.add_argument("--search-memory-mb", type=float, default=512)
    parser.add_argument("--profile-search", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.games, args.simulations, args.repeats, args.cpu_threads) < 1 or args.warmup < 0:
        parser.error("Games, simulations, repeats and threads must be positive; warmup nonnegative")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if "cuda" in args.backends and device.type != "cuda":
        parser.error("CUDA search requires a CUDA device")
    if args.mixed_precision == "fp16" and device.type != "cuda":
        parser.error("FP16 benchmarking requires CUDA")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if args.model:
        model, _ = load_checkpoint(args.model, device)
    else:
        model = AlphaZeroNet(AlphaZeroConfig(channels=args.channels, blocks=args.blocks)).to(device)
    model.eval()

    def collect(backend):
        context = torch.autocast("cuda", dtype=torch.float16) if args.mixed_precision == "fp16" else nullcontext()
        with context:
            return self_play_games(model, args.games, args.simulations, rng=np.random.default_rng(args.seed),
                                   search_backend=backend, search_memory_mb=args.search_memory_mb,
                                   profile_search=args.profile_search)

    for backend in args.backends:
        for _ in range(args.warmup):
            collect(backend)
        durations, position_counts = [], []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            start = time.perf_counter()
            _, metrics = collect(backend)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
            position_counts.append(metrics["positions"])
        record = dict(backend=backend, device=str(device), games=args.games, simulations=args.simulations,
                      mixed_precision=args.mixed_precision, repeats=args.repeats,
                      median_seconds=statistics.median(durations), seconds=durations,
                      games_per_minute=args.games * 60 / statistics.median(durations),
                      positions_per_second=sum(position_counts) / sum(durations),
                      positions_per_run=position_counts,
                      peak_cuda_memory_mib=torch.cuda.max_memory_allocated(device) / 1024**2
                      if device.type == "cuda" else None)
        if args.profile_search:
            record["last_run_search_profile"] = metrics["search_profile"]
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
