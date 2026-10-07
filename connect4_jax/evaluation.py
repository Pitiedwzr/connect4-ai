"""Fixed-opponent CPU evaluation, separate from self-play RNG and losses."""
import time
from dataclasses import asdict

import numpy as np

from .agent import AlphaZeroAgent


def score(counts):
    games = sum(counts.values())
    return (counts["wins"] + 0.5 * counts["draws"]) / games if games else 0.0


def evaluate(model, *, games=10, depths=(2, 4), seed=12345, opening_plies=4,
             simulations=128, search_policy="puct", c_puct=1.5,
             proven_win_priority=False, prior_temperature=1.0, gumbel_q_scale=0.1):
    # Baseline opponents use the existing shared paired-opening benchmark.
    # Import lazily: training without evaluation and CPU inference need no torch.
    from benchmark import evaluate_network
    if (model.config.rows, model.config.cols, model.config.connect) != (6, 7, 4):
        raise ValueError("Opponent evaluation requires standard 6x7 Connect 4")
    agent = AlphaZeroAgent(model, simulations, c_puct, search_policy,
                          proven_win_priority=proven_win_priority,
                          prior_temperature=prior_temperature, gumbel_q_scale=gumbel_q_scale)
    agent.warmup()  # Compilation excluded from move latency statistics.
    modes = {}
    for raw, name in ((True, "raw"), (False, "search")):
        times = []

        def move(board, my, other):
            start = time.perf_counter()
            action = agent.get_move(board, my, other, raw=raw)
            times.append(time.perf_counter() - start)
            return action

        results = evaluate_network(None, games, seed, depths, raw, opening_plies, move)
        aggregate = {"wins": 0, "draws": 0, "losses": 0}
        by_opponent = {}
        for opponent, seats in results.items():
            counts = {key: sum(seat[key] for seat in seats.values()) for key in aggregate}
            by_opponent[opponent] = score(counts)
            for key in aggregate:
                aggregate[key] += counts[key]
        modes[name] = dict(results=results, score=score(aggregate), scores_by_opponent=by_opponent,
                           move_latency_ms=dict(p50=float(np.percentile(times, 50) * 1000),
                                                p95=float(np.percentile(times, 95) * 1000)))
    opponent = f"minimax_{max(depths)}" if depths else "tactical_random"
    return dict(games_per_seat=games, depths=list(depths), seed=seed, opening_plies=opening_plies,
                simulations=simulations, search_policy=search_policy, search=asdict(agent.settings), modes=modes,
                selection_opponent=opponent,
                selection_score=modes["search"]["scores_by_opponent"][opponent],
                raw_selection_score=modes["raw"]["scores_by_opponent"][opponent])
