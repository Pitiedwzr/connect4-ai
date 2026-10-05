"""Data-parallel learner and independent device-local self-play trees."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .network import predict_batch
from .self_play import Trajectory, collect


def policy_value_loss(model, batch, dtype=jnp.float32):
    states, targets, outcomes = batch
    logits, values = predict_batch(model, states, dtype)
    legal = ~jnp.any(states[:, :, -1, :] != 0, axis=1)
    log_probs = jax.nn.log_softmax(jnp.where(legal, logits, jnp.finfo(jnp.float32).min), axis=-1)
    # Zero illegal entries before multiplication (also safe for -inf logits).
    policy_loss = -jnp.mean(jnp.sum(targets * jnp.where(legal, log_probs, 0.0), axis=-1))
    value_loss = jnp.mean(jnp.square(values - outcomes))
    return policy_loss + value_loss, jnp.stack((policy_loss, value_loss))


def make_optimizer(learning_rate=1e-3, weight_decay=1e-4):
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.adamw(learning_rate, weight_decay=weight_decay))


class Runner:
    """One process, any number of local CPU/GPU/TPU devices.

    Each device owns a batch of complete games. Search never crosses devices;
    only learner gradients are averaged. Global batch sizes are explicit.
    """
    def __init__(self, model, optimizer, devices, settings, games_per_device=64,
                 temperature_moves=10, dtype=jnp.float32):
        if not devices or games_per_device < 1 or temperature_moves < 0:
            raise ValueError("Devices/games must be positive and temperature moves nonnegative")
        self.mesh = Mesh(np.asarray(devices), ("devices",))
        self.count = len(devices)
        self.games = games_per_device * self.count
        self.params, self.static = eqx.partition(model, eqx.is_inexact_array)
        self.replicated = NamedSharding(self.mesh, P())
        self.batch_sharding = NamedSharding(self.mesh, P("devices"))
        self.params = jax.device_put(self.params, self.replicated)
        self.optimizer = optimizer
        self.optimizer_state = jax.device_put(optimizer.init(self.params), self.replicated)
        static = self.static

        def local_collect(params, keys):
            return collect(eqx.combine(params, static), keys[0], games_per_device,
                           settings, temperature_moves, dtype)

        specs = Trajectory(P(None, "devices"), P(None, "devices"),
                           P(None, "devices"), P(None, "devices"), P("devices"))
        self._collect = jax.jit(jax.shard_map(
            local_collect, mesh=self.mesh, in_specs=(P(), P("devices")),
            out_specs=specs, check_vma=False))

        def local_update(params, optimizer_state, batch):
            def loss_fn(weights):
                return policy_value_loss(eqx.combine(weights, static), batch, dtype)
            (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            grads = jax.lax.pmean(grads, "devices")
            metrics = jax.lax.pmean(jnp.concatenate((loss[None], components)), "devices")
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            return eqx.apply_updates(params, updates), optimizer_state, metrics

        self._update = jax.jit(jax.shard_map(
            local_update, mesh=self.mesh, in_specs=(P(), P(), P("devices")),
            out_specs=(P(), P(), P()), check_vma=False))

    @property
    def model(self):
        return eqx.combine(self.params, self.static)

    def restore_optimizer(self, state):
        self.optimizer_state = jax.device_put(state, self.replicated)

    def collect(self, key):
        # fold_in ties streams to logical device indices; no cloned self-play RNG.
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(self.count))
        return self._collect(self.params, jax.device_put(keys, self.batch_sharding))

    def update(self, batch):
        if len(batch[0]) % self.count:
            raise ValueError("Global batch size must be divisible by device count")
        batch = jax.device_put(tuple(batch), self.batch_sharding)
        self.params, self.optimizer_state, metrics = self._update(self.params, self.optimizer_state, batch)
        return metrics
