"""Fixed-opponent CPU evaluation, separate from self-play RNG and losses."""
import time
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from .agent import AlphaZeroAgent


def score(counts):
    games = sum(counts.values())
    return (counts["wins"] + 0.5 * counts["draws"]) / games if games else 0.0


def evaluate(model, *, games=10, depths=(2, 4), seed=12345, opening_plies=4,
             simulations=128, search_policy="puct", c_puct=1.5,
             proven_win_priority=False, prior_temperature=1.0, gumbel_q_scale=0.1,
             seeds=None, selection="deepest", opponent_model=None, opponent_sha256=None):
    # Baseline opponents use the existing shared paired-opening benchmark.
    # Import lazily: training without evaluation and CPU inference need no torch.
    from benchmark import evaluate_network
    if (model.config.rows, model.config.cols, model.config.connect) != (6, 7, 4):
        raise ValueError("Opponent evaluation requires standard 6x7 Connect 4")
    seeds = list(seeds) if seeds is not None else [seed]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Evaluation seeds must be nonempty, unique, and nonnegative")
    if selection not in ("deepest", "suite") or len(set(depths)) != len(depths):
        raise ValueError("Unknown selection rule or duplicate evaluation depths")
    agent = AlphaZeroAgent(model, simulations, c_puct, search_policy,
                          proven_win_priority=proven_win_priority,
                          prior_temperature=prior_temperature, gumbel_q_scale=gumbel_q_scale)
    agent.warmup()  # Compilation excluded from move latency statistics.
    frozen = None
    digest = None
    if opponent_model:
        digest = hashlib.sha256(Path(opponent_model).read_bytes()).hexdigest()
        if opponent_sha256 is not None and digest != opponent_sha256:
            raise ValueError("Frozen evaluation opponent changed since evaluation settings were recorded")
        frozen = AlphaZeroAgent.from_checkpoint(
            opponent_model, simulations, c_puct, search_policy,
            proven_win_priority=proven_win_priority,
            prior_temperature=prior_temperature, gumbel_q_scale=gumbel_q_scale)
        if frozen.model.config.rows != 6 or frozen.model.config.cols != 7 or frozen.model.config.connect != 4:
            raise ValueError("Frozen evaluation opponent requires standard 6x7 Connect 4")
        frozen.warmup()
    modes = {}
    for raw, name in ((True, "raw"), (False, "search")):
        times = []

        def move(board, my, other):
            start = time.perf_counter()
            action = agent.get_move(board, my, other, raw=raw)
            times.append(time.perf_counter() - start)
            return action

        results = {}
        per_seed = []
        extras = {"frozen_model": lambda board, my, other: frozen.get_move(board, my, other, raw=raw)} if frozen else None
        for current_seed in seeds:
            current = evaluate_network(None, games, current_seed, depths, raw, opening_plies, move, extras)
            per_seed.append(dict(seed=current_seed, results=current))
            for opponent, seats in current.items():
                total = results.setdefault(opponent, {seat: dict(wins=0, draws=0, losses=0) for seat in seats})
                for seat, counts in seats.items():
                    for key, count in counts.items():
                        total[seat][key] += count
        aggregate = {"wins": 0, "draws": 0, "losses": 0}
        by_opponent = {}
        for opponent, seats in results.items():
            counts = {key: sum(seat[key] for seat in seats.values()) for key in aggregate}
            by_opponent[opponent] = score(counts)
            for key in aggregate:
                aggregate[key] += counts[key]
        modes[name] = dict(results=results, score=score(aggregate), scores_by_opponent=by_opponent,
                           per_seed=per_seed,
                           move_latency_ms=dict(p50=float(np.percentile(times, 50) * 1000),
                                                p95=float(np.percentile(times, 95) * 1000)))
    opponent = f"minimax_{max(depths)}" if depths else "tactical_random"
    selected = ([f"minimax_{depth}" for depth in depths] or ["tactical_random"]) if selection == "suite" else [opponent]
    if selection == "suite" and frozen:
        selected.append("frozen_model")
    return dict(games_per_seat=games, depths=list(depths), seed=seed, seeds=seeds, opening_plies=opening_plies,
                simulations=simulations, search_policy=search_policy, search=asdict(agent.settings), modes=modes,
                opponent_model=str(opponent_model) if opponent_model else None, opponent_sha256=digest,
                selection_rule=selection, selection_opponent=opponent if selection == "deepest" else "suite",
                selection_opponents=selected,
                selection_score=float(np.mean([modes["search"]["scores_by_opponent"][key] for key in selected])),
                weakest_opponent_score=min(modes["search"]["scores_by_opponent"][key] for key in selected),
                raw_selection_score=float(np.mean([modes["raw"]["scores_by_opponent"][key] for key in selected])))


def evaluation_rank(report):
    """Rank suite mean first, then weakest opponent, then the raw-policy mean."""
    return (report["selection_score"], report.get("weakest_opponent_score", report["selection_score"]),
            report["raw_selection_score"])
