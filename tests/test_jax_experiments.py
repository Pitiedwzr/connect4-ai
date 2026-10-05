"""Search semantics, legal prefixes, retained checkpoints, and fixed evaluation."""
import json
import contextlib
import io
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from connect4_jax.agent import AlphaZeroAgent
from connect4_jax.checkpoint import load_checkpoint, save_checkpoint
from connect4_jax.config import Config
from connect4_jax.environment import from_board
from connect4_jax.evaluation import evaluate
from connect4_jax.network import PolicyValueNet
from connect4_jax.search import SearchConfig, search, tree_memory_bytes
from connect4_jax.self_play import collect, initial_positions
from test_jax import FixedNet, JaxTestCase, batch_state, history
from train_jax import parse_args, train

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(JaxTestCase):
    def test_gumbel_explores_low_prior_winning_action_and_masks_full_columns(self):
        model = FixedNet()
        model = eqx.tree_at(lambda m: m.logits, model, jnp.log(jnp.asarray([0.2, 0.2, 0.2, 1e-6, 0.2, 0.1, 0.1])))
        state = history([0, 6, 1, 6, 2, 5])
        settings = SearchConfig(64, policy="gumbel")
        run = eqx.filter_jit(search)
        output = run(model, batch_state(state), jax.random.PRNGKey(0), settings)
        self.assertTrue((np.asarray(output.search_tree.summary().visit_counts[0]) > 0).all())
        # Root coverage does not guarantee selection under an extreme prior.
        # With ordinary priors the same exact winning transition must win.
        ordinary = run(FixedNet(), batch_state(state), jax.random.PRNGKey(0), settings)
        self.assertEqual(int(ordinary.action[0]), 3)
        actual_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(output.search_tree))
        self.assertEqual(actual_bytes, tree_memory_bytes(model.config, 1, 64, "gumbel"))
        full = history([0] * 6)
        output = run(model, batch_state(full), jax.random.PRNGKey(0), settings)
        self.assertEqual(float(output.action_weights[0, 0]), 0)
        self.assertNotEqual(int(output.action[0]), 0)
        repeated = run(model, batch_state(full), jax.random.PRNGKey(123), settings)
        np.testing.assert_array_equal(output.action_weights, repeated.action_weights)
        self.assertEqual(int(output.action[0]), int(repeated.action[0]))

    def test_agent_uses_gumbel_recommended_action_and_checkpoint_policy(self):
        model = PolicyValueNet(Config(channels=8, blocks=0), jax.random.PRNGKey(0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gumbel.eqx"
            save_checkpoint(path, model, metadata={"search": {"policy": "gumbel"}})
            agent = AlphaZeroAgent.from_checkpoint(path, simulations=8)
            self.assertEqual(agent.settings.policy, "gumbel")
            visits = jnp.asarray([7, 1, 0, 0, 0, 0, 0], jnp.float32)
            with patch("connect4_jax.agent._search", return_value=(visits / 8, visits, jnp.float32(0), jnp.int32(1))):
                self.assertEqual(agent.get_move(np.zeros((6, 7))), 1)
            override = AlphaZeroAgent.from_checkpoint(path, simulations=8, search_policy="puct")
            self.assertEqual(override.settings.policy, "puct")

    def test_prefixes_obey_gravity_perspective_and_exclude_terminal_positions(self):
        config = Config()
        states = eqx.filter_jit(initial_positions)(config, jax.random.PRNGKey(9), 128, 1.0, 8)
        arrays = jax.device_get(states)
        lengths = arrays.heights.sum(axis=-1)
        self.assertTrue((lengths <= 8).all())
        self.assertGreater(int((lengths > 0).sum()), 100)
        self.assertFalse(arrays.done.any())
        np.testing.assert_array_equal(arrays.to_play, 1 + lengths % 2)
        for board, player in zip(arrays.board, arrays.to_play):
            validated = from_board(board, int(player), config)
            self.assertFalse(bool(validated.done))
        repeated = eqx.filter_jit(initial_positions)(config, jax.random.PRNGKey(9), 128, 1.0, 8)
        np.testing.assert_array_equal(states.board, repeated.board)
        tiny = Config(rows=1, cols=3, connect=3)
        states = eqx.filter_jit(initial_positions)(tiny, jax.random.PRNGKey(4), 128, 1.0, 3)
        self.assertFalse(np.asarray(states.done).any())
        self.assertTrue((np.asarray(states.heights).sum(axis=-1) < 3).all())

    def test_prefixed_self_play_uses_actual_player_for_outcomes(self):
        config = Config(rows=4, cols=4, connect=3)
        trajectory = jax.device_get(eqx.filter_jit(collect)(
            FixedNet(config), jax.random.PRNGKey(11), 8, SearchConfig(8, policy="gumbel"),
            opening_fraction=1.0, opening_plies=4))
        self.assertTrue((trajectory.opening_plies > 0).any())
        for game in range(8):
            players = 1 + (trajectory.opening_plies[game] + np.arange(16)) % 2
            valid = trajectory.valid[:, game]
            expected = np.where(trajectory.winners[game] == 0, 0,
                                np.where(players == trajectory.winners[game], 1, -1))
            np.testing.assert_array_equal(trajectory.outcomes[valid, game], expected[valid])
            self.assertEqual(int(trajectory.states[0, game].sum()), int(trajectory.opening_plies[game]))
            self.assertLessEqual(int(valid.sum()) + int(trajectory.opening_plies[game]), 16)

    def test_presets_are_explicit_overrides_for_resume(self):
        args = parse_args(["--preset", "improve", "--resume", "old.eqx", "--simulations", "64"])
        self.assertEqual(args.search_policy, "gumbel")
        self.assertEqual(args.learning_rate, 0.0003)
        self.assertEqual(args.opening_fraction, 0.2)
        self.assertEqual(args.simulations, 64)
        self.assertTrue({"search_policy", "learning_rate", "opening_fraction"} <= args.explicit_options)
        larger = parse_args(["--preset", "larger"])
        self.assertEqual((larger.channels, larger.blocks), (64, 2))

    def test_best_checkpoint_keeps_strongest_evaluation_across_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, best, snapshots = root / "latest.eqx", root / "best.eqx", root / "history"
            args = parse_args(["--cpu", "--iterations", "2", "--games-per-device", "2",
                               "--search-policy", "gumbel", "--simulations", "8", "--batch-size", "4",
                               "--warmup-positions", "4", "--updates-per-iteration", "1", "--channels", "8",
                               "--blocks", "0", "--replay-capacity", "128", "--eval-every", "1",
                               "--eval-games", "1", "--eval-depths", "--eval-simulations", "8",
                               "--checkpoint-every", "1", "--checkpoint-dir", str(snapshots),
                               "--output", str(output), "--best-output", str(best)])
            reports = [{"selection_score": 0.8, "raw_selection_score": 0.5},
                       {"selection_score": 0.6, "raw_selection_score": 0.5}]
            with patch("connect4_jax.evaluation.evaluate", side_effect=reports), contextlib.redirect_stdout(io.StringIO()):
                train(args)
            selected, selected_metadata = load_checkpoint(best)
            first, _ = load_checkpoint(snapshots / "latest.iter-0000001.eqx")
            self.assertEqual(selected_metadata["iteration"], 1)
            for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(selected)):
                np.testing.assert_array_equal(a, b)
            saved = best.read_bytes()
            resumed = parse_args(["--cpu", "--resume", str(output), "--iterations", "1"])
            with patch("connect4_jax.evaluation.evaluate", return_value={"selection_score": 0.7, "raw_selection_score": 0.5}), contextlib.redirect_stdout(io.StringIO()):
                train(resumed)
            self.assertEqual(best.read_bytes(), saved)
            _, metadata = load_checkpoint(output)
            self.assertEqual(metadata["best_evaluation"]["iteration"], 1)
            reports = list((root / "latest_evaluations").glob("*.json"))
            self.assertEqual(len(reports), 3)

    def test_evaluation_records_both_modes_and_preserves_random_state(self):
        model = PolicyValueNet(Config(channels=8, blocks=0), jax.random.PRNGKey(2))
        random.seed(42)
        previous = random.getstate()
        result = evaluate(model, games=1, depths=(), simulations=8, search_policy="gumbel")
        self.assertEqual(random.getstate(), previous)
        self.assertEqual(set(result["modes"]), {"raw", "search"})
        self.assertEqual(result["selection_opponent"], "tactical_random")
        self.assertTrue(0 <= result["selection_score"] <= 1)
        for mode in result["modes"].values():
            self.assertEqual(set(mode["results"]), {"random", "tactical_random"})
            self.assertGreater(mode["move_latency_ms"]["p50"], 0)

    def test_retained_checkpoints_and_gumbel_prefix_resume_match_full_run(self):
        command = [sys.executable, str(ROOT / "train_jax.py"), "--cpu", "--games-per-device", "2",
                   "--search-policy", "gumbel", "--simulations", "8", "--opening-fraction", "0.5",
                   "--opening-plies", "4", "--batch-size", "4", "--updates-per-iteration", "1",
                   "--warmup-positions", "4", "--replay-capacity", "64", "--rows", "4", "--cols", "4",
                   "--connect", "3", "--channels", "8", "--blocks", "0", "--checkpoint-every", "1"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full, partial = root / "full.eqx", root / "partial.eqx"
            snapshots = root / "history"
            for extra in (["--iterations", "2", "--output", str(full)],
                          ["--iterations", "1", "--output", str(partial), "--checkpoint-dir", str(snapshots), "--keep-checkpoints", "1"]):
                result = subprocess.run(command + extra, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list(snapshots.glob("*.eqx"))), 1)
            unrelated = snapshots / "other.eqx"
            unrelated.write_bytes(b"retain unrelated artifacts")
            result = subprocess.run([sys.executable, str(ROOT / "train_jax.py"), "--cpu", "--iterations", "1",
                                     "--resume", str(partial)], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(unrelated.read_bytes(), b"retain unrelated artifacts")
            self.assertEqual([p.name for p in snapshots.glob("partial.iter-*.eqx")], ["partial.iter-0000002.eqx"])
            full_model, full_metadata = load_checkpoint(full)
            partial_model, partial_metadata = load_checkpoint(partial)
            self.assertEqual(full_metadata["jax_key"], partial_metadata["jax_key"])
            self.assertEqual(full_metadata["numpy_rng"], partial_metadata["numpy_rng"])
            self.assertEqual(partial_metadata["search"]["policy"], "gumbel")
            for a, b in zip(jax.tree.leaves(full_model), jax.tree.leaves(partial_model)):
                np.testing.assert_array_equal(a, b)
