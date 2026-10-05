"""Runs in a fresh process with two simulated CPU devices."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from connect4_jax.config import Config
from connect4_jax.network import PolicyValueNet
from connect4_jax.search import SearchConfig
from connect4_jax.self_play import collect
from connect4_jax.training import Runner, make_optimizer


def main():
    devices = jax.devices("cpu")
    assert len(devices) == 2, devices
    config = Config(rows=4, cols=4, connect=3, channels=8, blocks=0)
    model = PolicyValueNet(config, jax.random.PRNGKey(42))
    settings = SearchConfig(2)
    parallel = Runner(model, make_optimizer(), devices, settings, games_per_device=2)
    serial = Runner(model, make_optimizer(), devices[:1], settings, games_per_device=2)
    rng = np.random.default_rng(42)
    states = rng.integers(0, 2, (8, 2, 4, 4)).astype(np.float32)
    states[:, :, -1, :] = 0  # all columns legal
    policies = rng.random((8, 4)).astype(np.float32)
    policies /= policies.sum(axis=-1, keepdims=True)
    outcomes = rng.choice([-1.0, 0.0, 1.0], 8).astype(np.float32)
    batch = (states, policies, outcomes)
    expected = np.asarray(serial.update(batch))
    actual = np.asarray(parallel.update(batch))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    for a, b in zip(jax.tree.leaves(serial.params), jax.tree.leaves(parallel.params)):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-6)
    key = jax.random.PRNGKey(7)
    trajectory = jax.device_get(parallel.collect(key))
    reference_collect = eqx.filter_jit(collect)
    for i in range(2):
        expected = jax.device_get(reference_collect(parallel.model, jax.random.fold_in(key, i), 2, settings))
        for name in ("states", "policies", "outcomes", "valid"):
            actual = getattr(trajectory, name)[:, i * 2:(i + 1) * 2]
            np.testing.assert_array_equal(actual, getattr(expected, name))
        np.testing.assert_array_equal(trajectory.winners[i * 2:(i + 1) * 2], expected.winners)
    print("Two-device gradients match the global batch; self-play matches independent RNG streams")


if __name__ == "__main__":
    main()
