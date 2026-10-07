"""Exact root-win proofs, corrected learning targets, and portable controls."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from connect4_jax.agent import AlphaZeroAgent
from alphazero import AlphaZeroConfig, Position
from connect4_jax.checkpoint import save_checkpoint
from connect4_jax.cli import add_search_arguments, search_overrides
from connect4_jax.config import Config
from connect4_jax.environment import empty, from_board, legal_actions
from connect4_jax.network import PolicyValueNet
from connect4_jax.replay import Replay
from connect4_jax.search import SearchConfig, prioritize_proven_wins, recurrent, search
from connect4_jax.self_play import collect
from test_jax import FixedNet, JaxTestCase, batch_state, history
from train_jax import parse_args

ROOT = Path(__file__).resolve().parents[1]


class ProofTests(JaxTestCase):
    def test_extreme_prior_win_corrects_action_and_training_target(self):
        model = eqx.tree_at(lambda m: m.logits, FixedNet(),
                           jnp.asarray([0., 0., 0., -1000., 0., 0., 0.]))
        states = batch_state(history([0, 6, 1, 6, 2, 5]))
        run = eqx.filter_jit(search)
        baseline = run(model, states, jax.random.PRNGKey(0), SearchConfig(64, policy="gumbel"))
        result = run(model, states, jax.random.PRNGKey(0),
                     SearchConfig(64, policy="gumbel", proven_win_priority=True))
        self.assertNotEqual(int(baseline.action[0]), 3)
        self.assertEqual(int(result.action[0]), 3)
        np.testing.assert_array_equal(result.action_weights[0], [0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(result.search_tree.summary().visit_counts,
                                      baseline.search_tree.summary().visit_counts)

    def test_multiple_wins_underflow_and_mixed_terminal_batch(self):
        model = eqx.tree_at(lambda m: m.logits, FixedNet(value=1),
                           jnp.asarray([-1000., 0., 0., 0., -1001., 0., 0.]))
        states = jax.tree.map(lambda *items: jnp.stack(items),
                             history([1, 6, 2, 6, 3, 5]), empty(model.config),
                             history([0, 1, 0, 1, 0, 1, 0]))
        run = eqx.filter_jit(search)
        base = run(model, states, jax.random.PRNGKey(0), SearchConfig(64, policy="gumbel"))
        result = run(model, states, jax.random.PRNGKey(0),
                     SearchConfig(64, policy="gumbel", proven_win_priority=True))
        self.assertTrue(np.isfinite(np.asarray(result.action_weights)).all())
        np.testing.assert_allclose(result.action_weights.sum(axis=-1), 1, atol=1e-6)
        self.assertEqual(int(result.action[0]), 0)
        np.testing.assert_allclose(result.action_weights[0, jnp.asarray([0, 4])],
                                   jax.nn.softmax(jnp.asarray([0., -1.])), atol=1e-6)
        np.testing.assert_array_equal(result.action_weights[0, jnp.asarray([1, 2, 3, 5, 6])], 0)
        # Even a saturated neural value cannot prove a nonterminal root edge.
        np.testing.assert_array_equal(result.action_weights[1:], base.action_weights[1:])
        np.testing.assert_array_equal(result.action[1:], base.action[1:])

    def test_unvisited_illegal_or_inactive_edges_cannot_prove_win(self):
        model = FixedNet()
        states = batch_state(empty(model.config))
        result = eqx.filter_jit(search)(model, states, jax.random.PRNGKey(0), SearchConfig(1))
        tree = result.search_tree
        unvisited = int(np.flatnonzero(np.asarray(tree.children_visits[0, 0]) == 0)[0])
        fake = tree.replace(children_rewards=tree.children_rewards.at[0, 0, unvisited].set(1),
                            children_discounts=tree.children_discounts.at[0, 0, unvisited].set(0))
        original = result.replace(search_tree=fake)
        legal = jax.vmap(legal_actions)(states)
        checked = prioritize_proven_wins(original, legal, states.done)
        np.testing.assert_array_equal(checked.action_weights, original.action_weights)
        fake = fake.replace(children_visits=fake.children_visits.at[0, 0, unvisited].set(1))
        original = result.replace(search_tree=fake)
        for allowed, done in ((legal.at[0, unvisited].set(False), states.done),
                              (legal, jnp.asarray([True]))):
            checked = prioritize_proven_wins(original, allowed, done)
            np.testing.assert_array_equal(checked.action_weights, original.action_weights)

    def test_puct_agent_respects_proven_target_and_no_win_visit_choice(self):
        model = FixedNet()
        agent = AlphaZeroAgent(model, 64, search_policy="puct", proven_win_priority=True)
        state = history([1, 6, 2, 6, 3, 5])
        self.assertIn(agent.get_move(np.asarray(state.board), int(state.to_play), 3-int(state.to_play)), [0, 4])
        np.testing.assert_array_equal(agent.last_result["policy"][[1, 2, 3, 5, 6]], 0)
        baseline = AlphaZeroAgent(model, 64, search_policy="puct")
        board = np.zeros((6, 7), np.int8)
        self.assertEqual(agent.get_move(board), baseline.get_move(board))
        np.testing.assert_array_equal(agent.last_result["policy"], baseline.last_result["policy"])

    def test_self_play_targets_label_only_proved_winning_moves(self):
        config = Config(rows=4, cols=4, connect=3)
        model = FixedNet(config)
        settings = SearchConfig(8, policy="gumbel", proven_win_priority=True,
                                prior_temperature=1.5, gumbel_q_scale=0.2)
        trajectory = jax.device_get(eqx.filter_jit(collect)(model, jax.random.PRNGKey(11), 8, settings,
                                                          opening_fraction=1., opening_plies=4))
        checked = 0
        for ply, game in np.argwhere(trajectory.valid):
            encoding = trajectory.states[ply, game]
            piece = 1 + int((trajectory.opening_plies[game]+ply) % 2)
            board = encoding[0]*piece + encoding[1]*(3-piece)
            position = Position.from_board(board, piece, AlphaZeroConfig(rows=4, cols=4, connect=3))
            winning = np.asarray([a in position.legal_moves() and position.play(a).winner == piece
                                  for a in range(4)])
            if winning.any():
                policy = trajectory.policies[ply, game]
                np.testing.assert_array_equal(policy[~winning], 0)
                self.assertAlmostEqual(float(policy.sum()), 1., places=6)
                checked += 1
        self.assertGreater(checked, 0)
        replay = Replay(128, config)
        replay.add(trajectory)  # Requires finite, normalized, legal labels.

    def test_controls_validate_restore_and_allow_explicit_disable(self):
        for name in ("prior_temperature", "gumbel_q_scale"):
            for value in (0., -1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    SearchConfig(**{name: value})
        settings = SearchConfig(8, policy="gumbel", proven_win_priority=True,
                                prior_temperature=1.5, gumbel_q_scale=0.2)
        metadata = {"search": asdict(settings)}
        self.assertEqual(SearchConfig.from_metadata(metadata), settings)
        self.assertFalse(SearchConfig.from_metadata(metadata, proven_win_priority=False).proven_win_priority)
        self.assertEqual(SearchConfig.from_metadata({}), SearchConfig())
        args = parse_args(["--resume", "old.eqx", "--no-proven-win-priority"])
        self.assertFalse(args.proven_win_priority)
        self.assertIn("proven_win_priority", args.explicit_options)
        parser = argparse.ArgumentParser()
        add_search_arguments(parser)
        add_search_arguments(parser, prefix="opponent-")
        args = parser.parse_args(["--proven-win-priority", "--no-opponent-proven-win-priority"])
        self.assertTrue(search_overrides(args)["proven_win_priority"])
        self.assertFalse(search_overrides(args, prefix="opponent_")["proven_win_priority"])

    def test_checkpoint_inference_controls_and_independent_benchmark_opponent(self):
        model = PolicyValueNet(Config(channels=8, blocks=0), jax.random.PRNGKey(2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.eqx"
            settings = SearchConfig(8, policy="gumbel", proven_win_priority=True,
                                    prior_temperature=1.5, gumbel_q_scale=0.2)
            save_checkpoint(path, model, metadata={"search": asdict(settings)})
            agent = AlphaZeroAgent.from_checkpoint(path, 8)
            self.assertEqual(agent.settings, settings)
            overridden = AlphaZeroAgent.from_checkpoint(path, 8, proven_win_priority=False, prior_temperature=1.)
            self.assertFalse(overridden.settings.proven_win_priority)
            self.assertEqual(overridden.settings.prior_temperature, 1.)
            self.assertEqual(overridden.settings.gumbel_q_scale, 0.2)
            old = Path(folder)/"old.eqx"
            save_checkpoint(old, model, metadata={"search": {"policy": "gumbel"}})
            self.assertFalse(AlphaZeroAgent.from_checkpoint(old, 8).settings.proven_win_priority)
            result = subprocess.run([sys.executable, str(ROOT/"benchmark.py"), "--agent", "equinox",
                "--model", str(old), "--simulations", "8", "--games", "1", "--depths",
                "--proven-win-priority", "--prior-temperature", "1.5", "--gumbel-q-scale", "0.2",
                "--opponent-model", str(old), "--opponent-simulations", "8",
                "--no-opponent-proven-win-priority"], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["search"]["proven_win_priority"])
            self.assertEqual(report["search"]["prior_temperature"], 1.5)
            self.assertFalse(report["opponent_search"]["proven_win_priority"])
            self.assertEqual(report["opponent_search"]["prior_temperature"], 1.)
            self.assertEqual(report["opponent_search"]["gumbel_q_scale"], 0.1)

    def test_recurrent_temperature_affects_leaves_and_preserves_exact_reward(self):
        model = eqx.tree_at(lambda m: m.logits, FixedNet(), jnp.arange(7, dtype=jnp.float32))
        state = batch_state(history([0, 6, 1, 6, 2, 5]))
        ordinary, _ = recurrent(model, jax.random.PRNGKey(0), jnp.asarray([4]), state)
        flattened, _ = recurrent(model, jax.random.PRNGKey(0), jnp.asarray([4]), state, prior_temperature=2.)
        np.testing.assert_array_equal(flattened.prior_logits, ordinary.prior_logits/2)
        winning, _ = recurrent(model, jax.random.PRNGKey(0), jnp.asarray([3]), state, prior_temperature=2.)
        self.assertEqual(float(winning.reward[0]), 1.)
        self.assertEqual(float(winning.discount[0]), 0.)
