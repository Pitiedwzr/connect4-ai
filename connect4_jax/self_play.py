"""Fixed-shape accelerator self-play, with no host work inside a game."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .environment import empty, encode, legal_actions, step
from .search import search


class Trajectory(NamedTuple):
    states: jax.Array
    policies: jax.Array
    outcomes: jax.Array
    valid: jax.Array
    winners: jax.Array
    opening_plies: jax.Array = None


def initial_positions(config, key, games, fraction, max_plies):
    """Uniform legal prefixes; discard any prefix that reaches a terminal state.

    Prefix moves receive no search labels and are excluded from replay. Rejection
    can make the realised fraction slightly lower than requested.
    """
    initial = jax.tree.map(lambda x: jnp.broadcast_to(x, (games,) + x.shape), empty(config))
    select_key, length_key, move_key = jax.random.split(key, 3)
    selected = jax.random.bernoulli(select_key, fraction, (games,))
    lengths = jnp.where(selected, jax.random.randint(length_key, (games,), 1, max_plies + 1), 0)

    def prefix_move(states, inputs):
        ply, rng = inputs
        legal = jax.vmap(legal_actions)(states)
        safe_logits = jnp.where(states.done[:, None], jnp.zeros_like(legal, dtype=jnp.float32),
                                jnp.where(legal, 0.0, -jnp.inf))
        actions = jax.random.categorical(rng, safe_logits)
        proposed = jax.vmap(lambda s, a: step(s, a, config))(states, actions)
        states = jax.tree.map(lambda new, old: jnp.where(
            (ply < lengths).reshape((games,) + (1,) * (old.ndim - 1)), new, old), proposed, states)
        return states, None

    states, _ = jax.lax.scan(prefix_move, initial,
                              (jnp.arange(max_plies), jax.random.split(move_key, max_plies)))
    return jax.tree.map(lambda new, old: jnp.where(
        states.done.reshape((games,) + (1,) * (old.ndim - 1)), old, new), states, initial)


def collect(model, key, games, settings, temperature_moves=10, dtype=jnp.float32,
            opening_fraction=0.0, opening_plies=8):
    config = model.config
    initial = jax.tree.map(lambda x: jnp.broadcast_to(x, (games,) + x.shape), empty(config))
    if opening_fraction:
        key, prefix_key = jax.random.split(key)
        initial = initial_positions(config, prefix_key, games, opening_fraction, opening_plies)

    def move(carry, ply):
        states, rng = carry
        rng, search_key, action_key = jax.random.split(rng, 3)

        def run_search(_):
            result = search(model, states, search_key, settings, add_noise=True, dtype=dtype)
            return result.action_weights, result.action.astype(jnp.int32)

        policies, proposed_actions = jax.lax.cond(jnp.any(~states.done), run_search,
                                lambda _: (jnp.zeros((games, config.cols), jnp.float32),
                                           jnp.zeros(games, jnp.int32)), operand=None)
        # Absorbing completed games have no replay target and use dummy action 0.
        acting_policy = jnp.where(states.done[:, None], jax.nn.one_hot(jnp.zeros(games, jnp.int32),
                                                                      config.cols), policies)
        sampled = jax.random.categorical(action_key, jnp.where(acting_policy > 0,
                                                              jnp.log(acting_policy), -jnp.inf))
        actions = jnp.where(jnp.sum(states.heights, axis=-1) < temperature_moves,
                             sampled, jnp.argmax(acting_policy, axis=-1)).astype(jnp.int32)
        if settings.policy == "gumbel":
            # Gumbel's recommended action is distinct from argmax(targets) or
            # argmax(visits). Gumbel noise supplies training exploration.
            actions = proposed_actions
        next_states = jax.vmap(lambda s, a: step(s, a, config))(states, actions)
        records = (jax.vmap(encode)(states).astype(jnp.uint8), policies, states.to_play, ~states.done)
        return (next_states, rng), records

    (final, _), (states, policies, players, valid) = jax.lax.scan(
        move, (initial, key), jnp.arange(config.rows * config.cols))
    outcomes = jnp.where(final.winner[None, :] == 0, 0.0,
                         jnp.where(players == final.winner[None, :], 1.0, -1.0))
    return Trajectory(states, policies, outcomes.astype(jnp.float32), valid, final.winner,
                      jnp.sum(initial.heights, axis=-1))
