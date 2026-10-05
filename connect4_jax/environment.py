"""Pure JAX Connect-N rules. Row zero is the bottom of the board."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config


class State(NamedTuple):
    board: jax.Array
    heights: jax.Array
    to_play: jax.Array
    winner: jax.Array
    done: jax.Array


def empty(config: Config) -> State:
    return State(jnp.zeros((config.rows, config.cols), jnp.int8),
                 jnp.zeros(config.cols, jnp.int32), jnp.int32(1),
                 jnp.int32(0), jnp.bool_(False))


def legal_actions(state: State):
    return (state.heights < state.board.shape[0]) & ~state.done


def has_connect(board, piece, connect):
    """Check runs without wraparound or fixed-width bitboard assumptions."""
    rows, cols = board.shape
    found = jnp.bool_(False)
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        height = rows - (connect - 1) * dr
        width = cols - (connect - 1) * abs(dc)
        if height <= 0 or width <= 0:
            continue
        run = jnp.ones((height, width), dtype=jnp.bool_)
        for i in range(connect):
            start_col = i if dc == 1 else connect - 1 - i if dc == -1 else 0
            start_row = i * dr
            run &= board[start_row:start_row + height, start_col:start_col + width] == piece
        found |= jnp.any(run)
    return found


def step(state: State, action, config: Config) -> State:
    """Apply a legal action; illegal and terminal transitions are absorbing.

    Host-facing callers validate actions. The absorbing fallback keeps dummy
    terminal leaves and masked completed games safe inside compiled search.
    """
    col = jnp.clip(action, 0, config.cols - 1)
    valid = (action >= 0) & (action < config.cols) & legal_actions(state)[col]
    row = jnp.minimum(state.heights[col], config.rows - 1)
    board = state.board.at[row, col].set(state.to_play.astype(jnp.int8))
    heights = state.heights.at[col].add(1)
    won = has_connect(board, state.to_play, config.connect)
    next_state = State(board, heights, 3 - state.to_play,
                       jnp.where(won, state.to_play, 0), won | jnp.all(heights == config.rows))
    return jax.tree.map(lambda new, old: jnp.where(valid, new, old), next_state, state)


def encode(state: State):
    return jnp.stack((state.board == state.to_play,
                      state.board == 3 - state.to_play)).astype(jnp.float32)


def from_board(board, to_play, config: Config) -> State:
    """Validate user input outside JIT; return device-independent array state."""
    board = np.asarray(board)
    if board.shape != (config.rows, config.cols) or to_play not in (1, 2):
        raise ValueError("Board shape or side to move does not match configuration")
    if not np.isin(board, (0, 1, 2)).all():
        raise ValueError("Board cells must be 0, 1 or 2")
    heights = np.count_nonzero(board, axis=0).astype(np.int32)
    if any(np.any(board[:h, c] == 0) or np.any(board[h:, c] != 0)
           for c, h in enumerate(heights)):
        raise ValueError("Pieces must obey gravity")
    wins = []
    for piece in (1, 2):
        won = False
        for r, c in np.argwhere(board == piece):
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                end_r, end_c = r + (config.connect - 1) * dr, c + (config.connect - 1) * dc
                if 0 <= end_r < config.rows and 0 <= end_c < config.cols:
                    won |= all(board[r + i * dr, c + i * dc] == piece
                               for i in range(config.connect))
        wins.append(won)
    if all(wins):
        raise ValueError("Both players cannot have winning lines")
    winner = 1 if wins[0] else 2 if wins[1] else 0
    return State(jnp.asarray(board, jnp.int8), jnp.asarray(heights),
                 jnp.int32(to_play), jnp.int32(winner),
                 jnp.bool_(winner != 0 or np.all(heights == config.rows)))
