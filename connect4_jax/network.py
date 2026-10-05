"""Compact residual policy/value CNN, with FP32 GroupNorm reductions."""
import math

import equinox as eqx
import jax
import jax.numpy as jnp

from .config import Config


def conv(layer, x, dtype):
    # Master weights remain FP32; only convolution operands use compute dtype.
    cast_layer = jax.tree.map(lambda a: a.astype(dtype) if eqx.is_array(a) else a, layer)
    return cast_layer(x.astype(dtype)).astype(jnp.float32)


class ResidualBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm

    def __init__(self, channels, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(channels, channels, 3, padding=1, use_bias=False, key=k1)
        self.conv2 = eqx.nn.Conv2d(channels, channels, 3, padding=1, use_bias=False, key=k2)
        self.norm1 = eqx.nn.GroupNorm(math.gcd(8, channels), channels)
        self.norm2 = eqx.nn.GroupNorm(math.gcd(8, channels), channels)

    def __call__(self, x, dtype):
        residual = jax.nn.relu(self.norm1(conv(self.conv1, x, dtype)))
        return jax.nn.relu(x + self.norm2(conv(self.conv2, residual, dtype)))


class PolicyValueNet(eqx.Module):
    config: Config = eqx.field(static=True)
    stem: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    blocks: tuple
    policy_conv: eqx.nn.Conv2d
    policy_fc: eqx.nn.Linear
    value_conv: eqx.nn.Conv2d
    value_fc: eqx.nn.Linear
    value_out: eqx.nn.Linear

    def __init__(self, config: Config, key):
        self.config = config
        keys = jax.random.split(key, config.blocks + 6)
        c = config.channels
        self.stem = eqx.nn.Conv2d(2, c, 3, padding=1, use_bias=False, key=keys[0])
        self.norm = eqx.nn.GroupNorm(math.gcd(8, c), c)
        self.blocks = tuple(ResidualBlock(c, keys[i + 1]) for i in range(config.blocks))
        self.policy_conv = eqx.nn.Conv2d(c, 2, 1, key=keys[-5])
        self.policy_fc = eqx.nn.Linear(2 * config.rows * config.cols, config.cols, key=keys[-4])
        self.value_conv = eqx.nn.Conv2d(c, 1, 1, key=keys[-3])
        self.value_fc = eqx.nn.Linear(config.rows * config.cols, 64, key=keys[-2])
        self.value_out = eqx.nn.Linear(64, 1, key=keys[-1])

    def __call__(self, x, dtype=jnp.float32):
        if x.shape != (2, self.config.rows, self.config.cols):
            raise ValueError("Input dimensions do not match model configuration")
        x = jax.nn.relu(self.norm(conv(self.stem, x, dtype)))
        for block in self.blocks:
            x = block(x, dtype)
        policy = self.policy_fc(jax.nn.relu(conv(self.policy_conv, x, dtype)).reshape(-1))
        value = jax.nn.relu(conv(self.value_conv, x, dtype)).reshape(-1)
        value = jnp.tanh(self.value_out(jax.nn.relu(self.value_fc(value))))[0]
        return policy.astype(jnp.float32), value.astype(jnp.float32)


def predict_batch(model, states, dtype=jnp.float32):
    return jax.vmap(lambda x: model(x, dtype))(states)
