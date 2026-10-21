"""Controlled multi-seed ranking, frozen opponents, and optional root diagnostics."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np

from connect4_jax.agent import AlphaZeroAgent
from connect4_jax.config import Config
from connect4_jax.evaluation import evaluate, evaluation_rank
from connect4_jax.network import PolicyValueNet
from connect4_jax.search import SearchConfig
from test_jax import FixedNet, JaxTestCase


class SuiteTests(JaxTestCase):
    def test_suite_balances_hard_opponents_and_aggregates_seeds(self):
        config = Config(channels=8, blocks=0)
        model = SimpleNamespace(config=config)
        fake = SimpleNamespace(settings=SearchConfig(8), model=model, warmup=lambda: None,
                               get_move=lambda *args, **kwargs: 3)

        def matches(net, games, seed, depths, raw, opening, move, extras):
            self.assertEqual(move(np.zeros((6, 7), np.int8), 1, 2), 3)
            self.assertEqual(extras['frozen_model'](np.zeros((6, 7), np.int8), 1, 2), 3)
            scores = {"random": 4, "tactical_random": 4, "minimax_4": 2 if seed == 11 else 4,
                      "minimax_5": 0 if seed == 11 else 2, "frozen_model": 4 if seed == 11 else 0}
            return {key: {"first": dict(wins=wins, draws=0, losses=4-wins),
                          "second": dict(wins=wins, draws=0, losses=4-wins)} for key, wins in scores.items()}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frozen.eqx'
            path.write_bytes(b'fixed fixture opponent')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('connect4_jax.evaluation.AlphaZeroAgent', return_value=fake) as agent_class, \
                 patch('benchmark.evaluate_network', side_effect=matches):
                agent_class.from_checkpoint.return_value = fake
                result = evaluate(model, games=4, seeds=[11, 22], depths=(4, 5), simulations=8,
                                  selection='suite', opponent_model=path, opponent_sha256=digest)
                agent_class.from_checkpoint.assert_called_once()
                self.assertEqual(result['selection_opponents'], ['minimax_4', 'minimax_5', 'frozen_model'])
                self.assertAlmostEqual(result['selection_score'], (0.75+0.25+0.5)/3)
                self.assertEqual(result['weakest_opponent_score'], 0.25)
                self.assertEqual(len(result['modes']['search']['per_seed']), 2)
                self.assertEqual(result['modes']['search']['results']['minimax_4']['first']['wins'], 6)
                self.assertEqual(result['opponent_sha256'], digest)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    evaluate(model, opponent_model=path, opponent_sha256='stale', simulations=8)
        # Mean wins over a narrow specialist; weakest opponent breaks a mean tie.
        balanced = dict(selection_score=0.8, weakest_opponent_score=0.7, raw_selection_score=0.5)
        specialist = dict(selection_score=0.75, weakest_opponent_score=0.75, raw_selection_score=1)
        self.assertGreater(evaluation_rank(balanced), evaluation_rank(specialist))
        self.assertGreater(evaluation_rank(balanced), evaluation_rank(dict(balanced, weakest_opponent_score=0.6)))

    def test_invalid_suite_seeds_and_depths_fail_before_inference(self):
        model = SimpleNamespace(config=Config())
        for seeds, depths in (([], (4,)), ([1, 1], (4,)), ([-1], (4,)), ([1], (4, 4))):
            with self.assertRaises(ValueError):
                evaluate(model, seeds=seeds, depths=depths)

    def test_opt_in_diagnostics_preserve_action_and_expose_exact_root_win(self):
        agent = AlphaZeroAgent(FixedNet(), simulations=8, search_policy='gumbel', proven_win_priority=True)
        board = np.zeros((6, 7), np.int8)
        for col, player in ((0, 1), (6, 2), (1, 1), (6, 2), (2, 1), (5, 2)):
            row = np.flatnonzero(board[:, col] == 0)[0]
            board[row, col] = player
        action = agent.get_move(board)
        ordinary = agent.last_result.copy()
        self.assertNotIn('qvalues', ordinary)
        self.assertEqual(agent.get_move(board, diagnostics=True), action)
        detailed = agent.last_result
        self.assertEqual(action, 3)
        self.assertTrue(detailed['proven_wins'][3])
        self.assertEqual(float(detailed['rewards'][3]), 1)
        self.assertEqual(float(detailed['discounts'][3]), 0)
        self.assertEqual(float(detailed['qvalues'][3]), 1)
        for name in ('policy', 'visits'):
            np.testing.assert_array_equal(ordinary[name], detailed[name])
        raw = agent.get_move(board, raw=True)
        self.assertEqual(agent.get_move(board, raw=True, diagnostics=True), raw)
        self.assertEqual(agent.last_result['mode'], 'raw')
        self.assertEqual(agent.last_result['network_logits'].shape, (7,))
