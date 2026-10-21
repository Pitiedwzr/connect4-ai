"""Tracing must preserve games and produce replayable, useful loss evidence."""
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from alphazero import Position
from analyze_traces import summarize
from benchmark import evaluate_network
from benchmark_traces import TraceWriter, tactical_diagnostics


class TraceTests(unittest.TestCase):
    def test_tracing_preserves_results_rng_and_replays_with_independent_rules(self):
        from game import get_valid_locations
        move = lambda board, my, other: get_valid_locations(board)[0]
        random.seed(57)
        state = random.getstate()
        plain = evaluate_network(None, games=3, seed=91, depths=(), opening_plies=4, move_fn=move)
        records = []
        traced = evaluate_network(None, games=3, seed=91, depths=(), opening_plies=4, move_fn=move,
                                  trace_fn=records.append, diagnostics_fn=lambda *args: {"qvalues": np.zeros(7)})
        self.assertEqual(plain, traced)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(records), 12)
        self.assertEqual(len({record['game_id'] for record in records}), 12)
        for record in records:
            position = Position.empty()
            for col in record['opening_actions']:
                position = position.play(col)
            np.testing.assert_array_equal(position.board(), record['initial_board'])
            self.assertEqual(position.to_play, record['initial_to_play'])
            for step in record['moves']:
                np.testing.assert_array_equal(position.board(), step['board_before'])
                self.assertEqual(position.to_play, step['player'])
                self.assertIn(step['action'], position.legal_moves())
                position = position.play(step['action'])
            self.assertEqual(position.winner, record['winner'])
            np.testing.assert_array_equal(position.board(), record['final_board'])
        paired = {(r['opponent'], r['game_index']): r['opening_id'] for r in records if r['candidate_seat'] == 1}
        for record in records:
            self.assertEqual(record['opening_id'], paired[record['opponent'], record['game_index']])

    def test_loss_and_seat_filters_and_exclusive_output(self):
        from game import get_valid_locations
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'losses.jsonl'
            with TraceWriter(path, {"model": "fixture"}, opponents=['random'], seat='second') as writer:
                evaluate_network(None, games=4, seed=14, depths=(),
                                 move_fn=lambda board, my, other: get_valid_locations(board)[0],
                                 trace_fn=writer, trace_filter=writer.eligible)
                written = writer.count
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]['type'], 'run')
            self.assertEqual(len(records)-1, written)
            self.assertGreater(written, 0)
            self.assertTrue(all(r['outcome'] == 'losses' and r['candidate_seat'] == 2 and r['opponent'] == 'random'
                                for r in records[1:]))
            result = summarize(path)
            self.assertEqual(result['counts']['losses'], written)
            self.assertTrue(result['mistakes'])
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                TraceWriter(path, {})
            self.assertEqual(original, path.read_bytes())

    def test_exact_tactics_distinguish_wins_and_avoidable_blocks(self):
        position = Position.empty()
        for action in [0, 6, 1, 6, 2, 5]:
            position = position.play(action)
        diagnostic = tactical_diagnostics(position.board(), 1, 4)
        self.assertEqual(diagnostic['immediate_wins'], [3])
        self.assertTrue(diagnostic['missed_immediate_win'])
        diagnostic = tactical_diagnostics(position.board(), 1, 3)
        self.assertFalse(diagnostic['allows_immediate_reply'])
        threat = Position.empty()
        for action in [6, 0, 6, 1, 5, 2]:
            threat = threat.play(action)
        diagnostic = tactical_diagnostics(threat.board(), 1, 4)
        self.assertEqual(diagnostic['safe_actions'], [3])
        self.assertTrue(diagnostic['avoidable_immediate_reply'])
