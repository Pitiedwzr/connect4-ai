"""Optional replayable JSONL benchmark traces; no model dependencies."""
import json
from pathlib import Path

import numpy as np


def tactical_diagnostics(board, player, action):
    """Exact one-ply facts; these do not classify deeper forced threats."""
    from game import drop_piece, get_next_open_row, get_valid_locations, winning_move
    wins, safe = [], []
    for col in get_valid_locations(board):
        after = board.copy()
        drop_piece(after, get_next_open_row(after, col), col, player)
        if winning_move(after, player):
            wins.append(int(col))
            safe.append(int(col))
            continue
        replies = []
        for reply in get_valid_locations(after):
            child = after.copy()
            drop_piece(child, get_next_open_row(child, reply), reply, 3-player)
            if winning_move(child, 3-player):
                replies.append(reply)
        if not replies:
            safe.append(int(col))
    return dict(immediate_wins=wins, safe_actions=safe,
                missed_immediate_win=bool(wins and action not in wins),
                allows_immediate_reply=action not in safe,
                avoidable_immediate_reply=bool(safe and action not in safe))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


class TraceWriter:
    def __init__(self, path, metadata, *, games="losses", opponents=None, seat="both"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        self.games, self.opponents, self.seat = games, set(opponents or ()), seat
        self.count = 0
        self.write(dict(type="run", schema_version=1, metadata=metadata,
                        filters=dict(games=games, opponents=sorted(self.opponents), seat=seat),
                        board_order="row 0 is bottom; columns and actions are zero based",
                        value_perspective="candidate to move; Q estimates on unvisited edges are not evidence"))

    def write(self, record):
        self.stream.write(json.dumps(record, default=json_value, allow_nan=False) + "\n")
        self.stream.flush()

    def eligible(self, opponent, seat):
        return ((not self.opponents or opponent in self.opponents)
                and (self.seat == "both" or self.seat == ("first" if seat == 1 else "second")))

    def __call__(self, record):
        if self.games == "all" or record["outcome"] == "losses":
            self.write(dict(type="game", **record))
            self.count += 1

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
