"""Measure CPU move latency and accelerator self-play throughput after warmup."""
import argparse
import json
import time
from dataclasses import asdict

import jax
import numpy as np

from connect4_jax.agent import AlphaZeroAgent
from connect4_jax.checkpoint import load_checkpoint
from connect4_jax.cli import add_search_arguments, search_overrides
from connect4_jax.config import Config
from connect4_jax.network import PolicyValueNet
from connect4_jax.search import SearchConfig
from connect4_jax.training import Runner, make_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Otherwise benchmark random 32-channel/2-block weights")
    parser.add_argument("--platform", choices=("cpu", "gpu", "tpu"), default="cpu")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--games-per-device", type=int, default=32)
    parser.add_argument("--simulations", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--collect", action="store_true", help="Measure full accelerator self-play instead of CPU moves")
    parser.add_argument("--search-policy", choices=("puct", "gumbel"), help="Defaults to checkpoint policy, or puct without weights")
    add_search_arguments(parser)
    parser.add_argument("--opening-fraction", type=float, default=0.0)
    parser.add_argument("--opening-plies", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.games_per_device < 1 or any(n < 1 for n in args.simulations):
        parser.error("Repeats, games and simulations must be positive")
    devices = jax.local_devices(backend=args.platform if args.collect else "cpu")
    if args.devices is not None:
        if not 1 <= args.devices <= len(devices):
            parser.error("Requested device count is unavailable")
        devices = devices[:args.devices]
    with jax.default_device(devices[0]):
        metadata = {}
        if args.model:
            model, metadata = load_checkpoint(args.model, devices[0])
        else:
            model = PolicyValueNet(Config(), jax.random.PRNGKey(42))
        reports = []
        for simulations in args.simulations:
            settings = SearchConfig.from_metadata(metadata, simulations=simulations,
                                                  policy=args.search_policy, **search_overrides(args))
            if args.collect:
                runner = Runner(model, make_optimizer(), devices, settings, args.games_per_device,
                                opening_fraction=args.opening_fraction, opening_plies=args.opening_plies)
                start = time.perf_counter()
                jax.block_until_ready(runner.collect(jax.random.PRNGKey(0)))
                warmup = time.perf_counter() - start
                seconds, positions = [], 0
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    trajectory = jax.device_get(runner.collect(jax.random.PRNGKey(repeat + 1)))
                    seconds.append(time.perf_counter() - start)
                    positions += int(trajectory.valid.sum())
                reports.append(dict(simulations=simulations, warmup_seconds=warmup,
                                    devices=len(devices), games_per_device=args.games_per_device,
                                    positions_per_second=positions / sum(seconds),
                                    games_per_minute=runner.games * args.repeats * 60 / sum(seconds)))
            else:
                agent = AlphaZeroAgent(model, simulations, settings.c_puct, settings.policy,
                                       proven_win_priority=settings.proven_win_priority,
                                       prior_temperature=settings.prior_temperature,
                                       gumbel_q_scale=settings.gumbel_q_scale)
                board = np.zeros((model.config.rows, model.config.cols), np.int8)
                start = time.perf_counter()
                agent.warmup()
                warmup = time.perf_counter() - start
                seconds = []
                # Representative legal prefixes, paired independently of search.
                from connect4_jax.environment import empty, legal_actions, step
                rng = np.random.default_rng(42)
                for repeat in range(args.repeats):
                    state = empty(model.config)
                    for _ in range(repeat % min(12, model.config.rows * model.config.cols)):
                        if bool(state.done):
                            break
                        state = step(state, int(rng.choice(np.flatnonzero(np.asarray(legal_actions(state))))), model.config)
                    if bool(state.done):
                        state = empty(model.config)
                    board, piece = np.asarray(state.board), int(state.to_play)
                    start = time.perf_counter()
                    agent.get_move(board, piece, 3 - piece)
                    seconds.append(time.perf_counter() - start)
                reports.append(dict(simulations=simulations, warmup_seconds=warmup,
                                    p50_ms=float(np.percentile(seconds, 50) * 1000),
                                    p95_ms=float(np.percentile(seconds, 95) * 1000)))
            reports[-1]["search"] = asdict(settings)
        print(json.dumps(dict(platform=args.platform if args.collect else "cpu",
                             search_policy=settings.policy, model=args.model,
                             mode="self_play" if args.collect else "move_latency", reports=reports), indent=2))


if __name__ == "__main__":
    main()
