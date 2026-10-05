"""Rule parity, mctx semantics, portable training, and CPU integration."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from alphazero import AlphaZeroConfig, AlphaZeroNet, Position
from connect4_jax.agent import AlphaZeroAgent
from connect4_jax.checkpoint import load_checkpoint, restore_training, save_checkpoint
from connect4_jax.config import Config
from connect4_jax.environment import empty, encode, from_board, legal_actions, step
from connect4_jax.network import PolicyValueNet, predict_batch
from connect4_jax.replay import Replay
from connect4_jax.search import SearchConfig, recurrent, search, tree_memory_bytes
from connect4_jax.self_play import collect
from connect4_jax.training import Runner, make_optimizer, policy_value_loss
from convert_checkpoint_jax import convert_model

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class FixedNet(eqx.Module):
    config: Config = eqx.field(static=True)
    logits: jax.Array
    value: jax.Array

    def __init__(self, config=Config(), value=0.0):
        self.config = config
        self.logits = jnp.zeros(config.cols)
        self.value = jnp.float32(value)

    def __call__(self, x, dtype=jnp.float32):
        return self.logits, self.value


def batch_state(state):
    return jax.tree.map(lambda x: x[None], state)


def history(columns, config=Config()):
    state = empty(config)
    for col in columns:
        state = step(state, col, config)
    return state


class JaxTestCase(unittest.TestCase):
    def tearDown(self):
        # Release compiled shapes before CLI tests start a second Python runtime.
        eqx.clear_caches()
        jax.clear_caches()
        gc.collect()


class EnvironmentTests(JaxTestCase):
    def test_random_games_match_reference_including_large_boards(self):
        rng = random.Random(42)
        for config in (Config(), Config(rows=8, cols=9), Config(rows=4, cols=4, connect=3)):
            advance = jax.jit(lambda s, a: step(s, a, config))
            for _ in range(12):
                reference = Position.empty(AlphaZeroConfig(**asdict(config)))
                state = empty(config)
                while reference.terminal_value() is None:
                    col = rng.choice(reference.legal_moves())
                    state = advance(state, col)
                    reference = reference.play(col)
                    np.testing.assert_array_equal(np.asarray(state.board), reference.board())
                    np.testing.assert_array_equal(np.asarray(encode(state)), reference.encode())
                    self.assertEqual(int(state.winner), reference.winner)
                    self.assertEqual(bool(state.done), reference.terminal_value() is not None)
                    self.assertEqual(np.flatnonzero(np.asarray(legal_actions(state))).tolist(), reference.legal_moves())

    def test_input_validation_and_absorbing_moves(self):
        config = Config()
        state = history([0] * 6)
        for invalid in (-1, 0, 7):
            after = step(state, invalid, config)
            for a, b in zip(state, after):
                np.testing.assert_array_equal(a, b)
        won = history([0, 6, 1, 6, 2, 5, 3])
        self.assertTrue(bool(won.done))
        np.testing.assert_array_equal(step(won, 4, config).board, won.board)
        malformed = np.zeros((6, 7), np.int8)
        malformed[1, 0] = 1
        with self.assertRaisesRegex(ValueError, "gravity"):
            from_board(malformed, 1, config)
        with self.assertRaises(ValueError):
            from_board(np.zeros((6, 7)), 3, config)


class SearchTests(JaxTestCase):
    def test_recurrent_signs_win_and_absorbing_terminal(self):
        model = FixedNet(value=0.75)
        transition, next_state = recurrent(model, jax.random.PRNGKey(0), jnp.asarray([3]), batch_state(empty(model.config)))
        self.assertEqual(float(transition.reward[0]), 0)
        self.assertEqual(float(transition.discount[0]), -1)
        self.assertAlmostEqual(float(transition.value[0]), 0.75)
        self.assertEqual(int(next_state.to_play[0]), 2)
        state = history([0, 6, 1, 6, 2, 5])
        transition, won = recurrent(model, jax.random.PRNGKey(0), jnp.asarray([3]), batch_state(state))
        self.assertEqual(float(transition.reward[0]), 1)
        self.assertEqual(float(transition.discount[0]), 0)
        self.assertEqual(float(transition.value[0]), 0)
        transition, again = recurrent(model, jax.random.PRNGKey(0), jnp.asarray([4]), won)
        self.assertEqual(float(transition.reward[0]), 0)
        self.assertTrue(np.isfinite(np.asarray(transition.prior_logits)).all())
        np.testing.assert_array_equal(won.board, again.board)

    def test_mctx_finds_win_and_block_and_masks_descendants(self):
        model = FixedNet()
        run = eqx.filter_jit(lambda s: search(model, batch_state(s), jax.random.PRNGKey(0), SearchConfig(96)))
        for columns, expected in (([0, 6, 1, 6, 2, 5], 3), ([6, 0, 6, 1, 5, 2], 3)):
            result = run(history(columns))
            self.assertEqual(int(np.argmax(np.asarray(result.action_weights[0]))), expected)
        state = history([0] * 6)
        result = run(state)
        self.assertEqual(float(result.action_weights[0, 0]), 0)
        tree = jax.device_get(result.search_tree)
        for node, action in np.argwhere(tree.children_index[0] >= 0):
            if tree.embeddings.done[0, node]:
                continue
            self.assertLess(int(tree.embeddings.heights[0, node, action]), model.config.rows)
        self.assertAlmostEqual(float(result.action_weights.sum()), 1.0, places=5)

    def test_draw_transition_uses_zero_discount(self):
        config = Config(rows=1, cols=3, connect=3)
        state = history([0, 1], config)
        output, final = recurrent(FixedNet(config), jax.random.PRNGKey(0), jnp.asarray([2]), batch_state(state))
        self.assertTrue(bool(final.done[0]))
        self.assertEqual(int(final.winner[0]), 0)
        self.assertEqual(float(output.reward[0]), 0)
        self.assertEqual(float(output.discount[0]), 0)

    def test_memory_estimate_matches_allocated_tree(self):
        model = FixedNet()
        result = eqx.filter_jit(search)(model, batch_state(empty(model.config)),
                                       jax.random.PRNGKey(0), SearchConfig(4))
        actual_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(result.search_tree))
        self.assertEqual(actual_bytes, tree_memory_bytes(model.config, 1, 4))


class LearningTests(JaxTestCase):
    def test_checkpoint_rejects_nonfinite_weights_without_replacing_previous(self):
        model = PolicyValueNet(Config(channels=8, blocks=0), jax.random.PRNGKey(0))
        bad = eqx.tree_at(lambda m: m.value_out.weight, model,
                          jnp.full_like(model.value_out.weight, jnp.nan))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.eqx"
            save_checkpoint(path, model)
            previous = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "finite"):
                save_checkpoint(path, bad)
            self.assertEqual(path.read_bytes(), previous)
            with zipfile.ZipFile(path) as bundle:
                import io
                stream = io.BytesIO()
                eqx.tree_serialise_leaves(stream, bad)
                # Write a deliberately corrupt replacement bundle, avoiding
                # the public saver that already rejects nonfinite parameters.
                metadata = bundle.read("metadata.json")
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr("metadata.json", metadata)
                bundle.writestr("model.eqx", stream.getvalue())
            with self.assertRaisesRegex(ValueError, "finite"):
                load_checkpoint(path)

    def test_gui_benchmark_and_inference_export(self):
        from ai_player import AlphaZeroPlayer
        from evaluator import PositionEvaluator
        model = PolicyValueNet(Config(channels=8, blocks=0), jax.random.PRNGKey(3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inference.eqx"
            save_checkpoint(path, model)
            board = np.zeros((6, 7), np.int8)
            player = AlphaZeroPlayer(model_path=path, simulations=2)
            self.assertIn(player.get_move(board), range(7))
            evaluator = PositionEvaluator(path, algorithm="alphazero")
            red, yellow, source = evaluator.evaluate(board, 1)
            self.assertAlmostEqual(red + yellow, 1)
            self.assertEqual(source, "AlphaZero value (Equinox)")
            for raw in (False, True):
                command = [sys.executable, str(ROOT / "benchmark.py"), "--agent", "equinox", "--model", str(path),
                           "--games", "1", "--depths", "--simulations", "2", "--cpu"]
                if raw:
                    command.append("--raw")
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["agent"], "equinox")
                self.assertEqual(report["device"], "cpu")
                self.assertEqual(set(report["results"]), {"random", "tactical_random"})
            replay = Replay(64, model.config)
            with self.assertRaisesRegex(ValueError, "Inference-only"):
                restore_training(path, (), replay)
            path.write_bytes(b"not a checkpoint")
            with self.assertRaises(ValueError):
                AlphaZeroAgent.from_checkpoint(path)
            fallback = PositionEvaluator(path, algorithm="alphazero")
            self.assertIsNone(fallback.equinox_agent)

    def test_cpu_inference_import_does_not_require_torch(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; from connect4_jax.agent import AlphaZeroAgent; assert 'torch' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_forward_conversion(self):
        legacy = AlphaZeroNet(AlphaZeroConfig(channels=8, blocks=1)).eval()
        converted = convert_model(legacy)
        inputs = np.random.default_rng(42).integers(0, 2, (4, 2, 6, 7)).astype(np.float32)
        with torch.no_grad():
            logits, values = legacy(torch.from_numpy(inputs))
        actual_logits, actual_values = eqx.filter_jit(predict_batch)(converted, jnp.asarray(inputs))
        np.testing.assert_allclose(actual_logits, logits.numpy(), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual_values, values.numpy().reshape(-1), atol=2e-5, rtol=2e-5)

    def test_self_play_outcomes_mirrors_and_learning(self):
        config = Config(rows=4, cols=4, connect=3, channels=8, blocks=0)
        model = PolicyValueNet(config, jax.random.PRNGKey(0))
        runner = Runner(model, make_optimizer(), [jax.devices("cpu")[0]], SearchConfig(4), games_per_device=2)
        trajectory = jax.device_get(runner.collect(jax.random.PRNGKey(2)))
        for game in range(2):
            valid = trajectory.valid[:, game]
            self.assertGreater(valid.sum(), 0)
            players = 1 + np.arange(config.rows * config.cols) % 2
            expected = np.where(trajectory.winners[game] == 0, 0,
                                np.where(players == trajectory.winners[game], 1, -1))
            np.testing.assert_array_equal(trajectory.outcomes[valid, game], expected[valid])
        replay = Replay(128, config)
        replay.add(trajectory)
        states, policies, outcomes = replay.sample(64, np.random.default_rng(3))
        legal = ~states[:, :, -1, :].any(axis=1)
        self.assertTrue((policies[~legal] == 0).all())
        original = {(s.tobytes(), p.tobytes(), float(z)) for s, p, z in
                    zip(replay.states[:len(replay)], replay.policies[:len(replay)], replay.outcomes[:len(replay)])}
        for s, p, z in zip(states.astype(np.uint8), policies, outcomes):
            self.assertTrue((s.tobytes(), p.tobytes(), float(z)) in original or
                            (s[:, :, ::-1].tobytes(), p[::-1].tobytes(), float(z)) in original)
        before = jax.tree.leaves(runner.params)[0].copy()
        metrics = np.asarray(runner.update((states[:4], policies[:4], outcomes[:4])))
        self.assertTrue(np.isfinite(metrics).all())
        self.assertFalse(np.array_equal(before, jax.tree.leaves(runner.params)[0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.eqx"
            save_checkpoint(path, runner.model, metadata={"iteration": 1},
                            optimizer_state=runner.optimizer_state, replay=replay)
            loaded, metadata = load_checkpoint(path, jax.devices("cpu")[0])
            self.assertEqual(metadata["iteration"], 1)
            for a, b in zip(jax.tree.leaves(runner.model), jax.tree.leaves(loaded)):
                np.testing.assert_array_equal(a, b)
            restored_replay = Replay(128, config)
            restored_opt = restore_training(path, runner.optimizer_state, restored_replay)
            for a, b in zip(jax.tree.leaves(runner.optimizer_state), jax.tree.leaves(restored_opt)):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(restored_replay.states[:len(replay)], replay.states[:len(replay)])
            agent = AlphaZeroAgent.from_checkpoint(path, simulations=4)
            self.assertEqual(agent.model.stem.weight.device.platform, "cpu")
            self.assertIn(agent.get_move(np.zeros((4, 4)), raw=True), range(4))
            self.assertIn(agent.get_move(np.zeros((4, 4))), range(4))

    def test_resume_matches_uninterrupted_training(self):
        command = [sys.executable, str(ROOT / "train_jax.py"), "--cpu", "--games-per-device", "2",
                   "--simulations", "2", "--batch-size", "4", "--updates-per-iteration", "1",
                   "--warmup-positions", "4", "--replay-capacity", "64", "--rows", "4", "--cols", "4",
                   "--connect", "3", "--channels", "8", "--blocks", "0"]
        with tempfile.TemporaryDirectory() as folder:
            full, partial = (Path(folder) / name for name in ("full.eqx", "partial.eqx"))
            for extra in (["--iterations", "2", "--output", str(full)],
                          ["--iterations", "1", "--output", str(partial)]):
                result = subprocess.run(command + extra, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([sys.executable, str(ROOT / "train_jax.py"), "--cpu", "--iterations", "1",
                                     "--resume", str(partial)], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            full_model, full_metadata = load_checkpoint(full)
            partial_model, partial_metadata = load_checkpoint(partial)
            self.assertEqual(full_metadata["jax_key"], partial_metadata["jax_key"])
            self.assertEqual(full_metadata["numpy_rng"], partial_metadata["numpy_rng"])
            self.assertEqual(full_metadata["games_played"], partial_metadata["games_played"])
            for a, b in zip(jax.tree.leaves(full_model), jax.tree.leaves(partial_model)):
                np.testing.assert_array_equal(a, b)
            with zipfile.ZipFile(full) as a, zipfile.ZipFile(partial) as b:
                self.assertEqual(a.read("optimizer.eqx"), b.read("optimizer.eqx"))
                self.assertEqual(a.read("replay.npz"), b.read("replay.npz"))


if __name__ == "__main__":
    unittest.main()
