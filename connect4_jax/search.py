"""mctx PUCT/Gumbel with exact Connect-N transitions and alternating-player backup."""
from dataclasses import dataclass, fields
from functools import partial
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
    policy: str = "puct"
    proven_win_priority: bool = False
    prior_temperature: float = 1.0
    gumbel_q_scale: float = 0.1

    def __post_init__(self):
        if self.policy not in ("puct", "gumbel"):
            raise ValueError("Search policy must be puct or gumbel")
        if self.simulations < 1 or self.c_puct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("Simulations, c-puct and Dirichlet alpha must be positive")
        if not math.isfinite(self.c_puct) or not math.isfinite(self.dirichlet_alpha):
            raise ValueError("Search constants must be finite")
        if not 0 <= self.noise_fraction <= 1:
            raise ValueError("noise_fraction must be in [0,1]")
        for name in ("prior_temperature", "gumbel_q_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @classmethod
    def from_metadata(cls, metadata, **overrides):
        """Load saved search settings; missing fields keep legacy defaults."""
        saved = metadata.get("search", {})
        training = metadata.get("training_args", {})
        values = {field.name: saved.get(field.name, training.get(
            "search_policy" if field.name == "policy" else field.name, field.default))
                  for field in fields(cls)}
        values.update({name: value for name, value in overrides.items() if value is not None})
        return cls(**values)


def tree_memory_bytes(config, games, simulations, policy="puct"):
    """Persistent mctx arrays, excluding NN activations and compiler temporaries."""
    state_bytes = config.rows * config.cols + 4 * config.cols + 9
    node_bytes = state_bytes + 20 + 24 * config.cols
    return games * ((simulations + 1) * node_bytes + config.cols
                    + (4 * config.cols if policy == "gumbel" else 0))


def masked_logits(logits, legal, done):
    # Terminal leaves use harmless finite logits. Zero discount isolates them
    # from the game; repeated terminal expansions remain absorbing.
    masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
    return jnp.where(done[:, None], jnp.zeros_like(masked), masked)


def recurrent(model, key, actions, states, dtype=jnp.float32, prior_temperature=1.0):
    del key
    next_states = jax.vmap(lambda s, a: step(s, a, model.config))(states, actions)
    logits, values = predict_batch(model, jax.vmap(encode)(next_states), dtype)
    if prior_temperature != 1.0:
        logits = logits / prior_temperature
    newly_won = ~states.done & next_states.done & (next_states.winner == states.to_play)
    reward = newly_won.astype(jnp.float32)
    discount = jnp.where(next_states.done, 0.0, -1.0).astype(jnp.float32)
    logits = masked_logits(logits, jax.vmap(legal_actions)(next_states), next_states.done)
    values = jnp.where(next_states.done, 0.0, values)
    return mctx.RecurrentFnOutput(reward=reward, discount=discount,
                                 prior_logits=logits, value=values), next_states


def prioritize_proven_wins(result, legal, done):
    """Correct action and target only for visited, exact terminal winning edges.

    A neural Q estimate of +1 is insufficient evidence. The tree's game reward
    and zero discount establish the proof. No extra expansion or inference runs.
    """
    tree = result.search_tree
    index = tree.ROOT_INDEX
    wins = ((tree.children_visits[:, index] > 0)
            & (tree.children_rewards[:, index] == 1)
            & (tree.children_discounts[:, index] == 0)
            & legal & ~done[:, None])
    has_win = jnp.any(wins, axis=-1)
    # Finite dummy logits avoid NaNs in batch slots with no proved win. Compute
    # from logits rather than underflowed action_weights, including tiny priors.
    logits = jnp.where(wins, tree.children_prior_logits[:, index], -jnp.inf)
    logits = jnp.where(has_win[:, None], logits, jnp.zeros_like(logits))
    targets = jax.nn.softmax(logits)
    action = jnp.argmax(logits, axis=-1).astype(result.action.dtype)
    return result.replace(action=jnp.where(has_win, action, result.action),
                          action_weights=jnp.where(has_win[:, None], targets, result.action_weights))


def search(model, states, key, settings: SearchConfig, *, add_noise=False, dtype=jnp.float32):
    """One tree per batch item. Rebuild trees at each actual game move."""
    legal = jax.vmap(legal_actions)(states)
    logits, values = predict_batch(model, jax.vmap(encode)(states), dtype)
    if settings.prior_temperature != 1.0:
        logits = logits / settings.prior_temperature
    root = mctx.RootFnOutput(prior_logits=masked_logits(logits, legal, states.done),
                             value=jnp.where(states.done, 0.0, values), embedding=states)
    # Completed batch slots need a dummy root action; never record their targets.
    invalid = ~jnp.where(states.done[:, None], jnp.ones_like(legal), legal)
    if settings.policy == "gumbel":
        if settings.simulations < model.config.cols:
            raise ValueError("Gumbel search requires at least cols simulations to consider every root action")
        result = mctx.gumbel_muzero_policy(
            model, key, root,
            lambda params, rng, action, embedding: recurrent(
                params, rng, action, embedding, dtype, settings.prior_temperature),
            num_simulations=settings.simulations, invalid_actions=invalid,
            max_depth=model.config.rows * model.config.cols,
            max_num_considered_actions=model.config.cols,
            gumbel_scale=1.0 if add_noise else 0.0,
            qtransform=partial(mctx.qtransform_completed_by_mix_value,
                               value_scale=settings.gumbel_q_scale))
    else:
        result = mctx.muzero_policy(
            model, key, root,
            lambda params, rng, action, embedding: recurrent(
                params, rng, action, embedding, dtype, settings.prior_temperature),
            num_simulations=settings.simulations, invalid_actions=invalid,
            max_depth=model.config.rows * model.config.cols,
            pb_c_init=settings.c_puct, pb_c_base=19652,
            dirichlet_alpha=settings.dirichlet_alpha,
            dirichlet_fraction=settings.noise_fraction if add_noise else 0.0,
            temperature=1.0)
    if settings.proven_win_priority:
        result = prioritize_proven_wins(result, legal, states.done)
    return result
