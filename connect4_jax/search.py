"""mctx PUCT with exact Connect-N transitions and alternating-player backup."""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import mctx

from .environment import encode, legal_actions, step
from .network import predict_batch


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    noise_fraction: float = 0.25

    def __post_init__(self):
        if self.simulations < 1 or self.c_puct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("Simulations, c-puct and Dirichlet alpha must be positive")
        if not math.isfinite(self.c_puct) or not math.isfinite(self.dirichlet_alpha):
            raise ValueError("Search constants must be finite")
        if not 0 <= self.noise_fraction <= 1:
            raise ValueError("noise_fraction must be in [0,1]")


def tree_memory_bytes(config, games, simulations):
    """Persistent mctx arrays, excluding NN activations and compiler temporaries."""
    state_bytes = config.rows * config.cols + 4 * config.cols + 9
    node_bytes = state_bytes + 20 + 24 * config.cols
    return games * ((simulations + 1) * node_bytes + config.cols)


def masked_logits(logits, legal, done):
    # Terminal leaves use harmless finite logits. Zero discount isolates them
    # from the game; repeated terminal expansions remain absorbing.
    masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
    return jnp.where(done[:, None], jnp.zeros_like(masked), masked)


def recurrent(model, key, actions, states, dtype=jnp.float32):
    del key
    next_states = jax.vmap(lambda s, a: step(s, a, model.config))(states, actions)
    logits, values = predict_batch(model, jax.vmap(encode)(next_states), dtype)
    newly_won = ~states.done & next_states.done & (next_states.winner == states.to_play)
    reward = newly_won.astype(jnp.float32)
    discount = jnp.where(next_states.done, 0.0, -1.0).astype(jnp.float32)
    logits = masked_logits(logits, jax.vmap(legal_actions)(next_states), next_states.done)
    values = jnp.where(next_states.done, 0.0, values)
    return mctx.RecurrentFnOutput(reward=reward, discount=discount,
                                 prior_logits=logits, value=values), next_states


def search(model, states, key, settings: SearchConfig, *, add_noise=False, dtype=jnp.float32):
    """One tree per batch item. Rebuild trees at each actual game move."""
    legal = jax.vmap(legal_actions)(states)
    logits, values = predict_batch(model, jax.vmap(encode)(states), dtype)
    root = mctx.RootFnOutput(prior_logits=masked_logits(logits, legal, states.done),
                             value=jnp.where(states.done, 0.0, values), embedding=states)
    # Completed batch slots need a dummy root action; never record their targets.
    invalid = ~jnp.where(states.done[:, None], jnp.ones_like(legal), legal)
    return mctx.muzero_policy(
        model, key, root,
        lambda params, rng, action, embedding: recurrent(params, rng, action, embedding, dtype),
        num_simulations=settings.simulations, invalid_actions=invalid,
        max_depth=model.config.rows * model.config.cols,
        pb_c_init=settings.c_puct, pb_c_base=19652,
        dirichlet_alpha=settings.dirichlet_alpha,
        dirichlet_fraction=settings.noise_fraction if add_noise else 0.0,
        temperature=1.0)
