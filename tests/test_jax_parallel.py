"""Validate sharding and all-reduce without requiring a GPU or TPU."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ParallelTests(unittest.TestCase):
    def test_two_device_gradients_and_game_streams(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2")
        result = subprocess.run([sys.executable, str(root / "tests/jax_parallel_worker.py")],
                                cwd=root, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
