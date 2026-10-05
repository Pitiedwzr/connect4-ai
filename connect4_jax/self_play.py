"""Fixed-shape accelerator self-play, with no host work inside a game."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .environment import empty, encode, step
from .search import search


class Trajectory(NamedTuple):
    states: jax.Array
    policies: jax.Array
    outcomes: jax.Array
    valid: jax.Array
    winners: jax.Array


def collect(model, key, games, settings, temperature_moves=10, dtype=jnp.float32):
    config = model.config
    initial = jax.tree.map(lambda x: jnp.broadcast_to(x, (games,) + x.shape), empty(config))

    def move(carry, ply):
        states, rng = carry
        rng, search_key, action_key = jax.random.split(rng, 3)

        def run_search(_):
            return search(model, states, search_key, settings, add_noise=True, dtype=dtype).action_weights

        policies = jax.lax.cond(jnp.any(~states.done), run_search,
                                lambda _: jnp.zeros((games, config.cols), jnp.float32), operand=None)
        # Absorbing completed games have no replay target and use dummy action 0.
        acting_policy = jnp.where(states.done[:, None], jax.nn.one_hot(jnp.zeros(games, jnp.int32),
                                                                      config.cols), policies)
        sampled = jax.random.categorical(action_key, jnp.where(acting_policy > 0,
                                                              jnp.log(acting_policy), -jnp.inf))
        actions = jnp.where(ply < temperature_moves, sampled, jnp.argmax(acting_policy, axis=-1))
        next_states = jax.vmap(lambda s, a: step(s, a, config))(states, actions)
        records = (jax.vmap(encode)(states).astype(jnp.uint8), policies, states.to_play, ~states.done)
        return (next_states, rng), records

    (final, _), (states, policies, players, valid) = jax.lax.scan(
        move, (initial, key), jnp.arange(config.rows * config.cols))
    outcomes = jnp.where(final.winner[None, :] == 0, 0.0,
                         jnp.where(players == final.winner[None, :], 1.0, -1.0))
    return Trajectory(states, policies, outcomes.astype(jnp.float32), valid, final.winner)
