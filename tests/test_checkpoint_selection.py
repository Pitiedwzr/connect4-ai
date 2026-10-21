"""Offline selection must preserve candidates and rank a common fixed suite."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from select_jax_checkpoint import select_checkpoints


class SelectionTests(unittest.TestCase):
    def test_common_suite_winner_metadata_and_stable_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / f'{name}.eqx' for name in ('first', 'second', 'third')]
            for path in paths:
                path.write_bytes(path.name.encode())
            frozen = root / 'frozen.eqx'
            frozen.write_bytes(b'frozen')
            output, report = root / 'winner.eqx', root / 'report.json'
            original = [path.read_bytes() for path in paths]
            evaluations = [dict(selection_score=score, weakest_opponent_score=0.5,
                                raw_selection_score=0.5, search={'policy': 'gumbel', 'proven_win_priority': True})
                           for score in (0.6, 0.8, 0.8)]
            with patch('connect4_jax.checkpoint.load_checkpoint', side_effect=[(name, {'iteration': 10+i}) for i, name in enumerate(('a', 'b', 'c'))]), \
                 patch('connect4_jax.evaluation.evaluate', side_effect=evaluations) as evaluate, \
                 patch('connect4_jax.checkpoint.save_checkpoint') as save, contextlib.redirect_stdout(io.StringIO()):
                selected = select_checkpoints(paths, report_path=report, best_output=output,
                                              seeds=[17, 18], depths=[4, 5], opponent_model=frozen)
            self.assertEqual(selected['winner']['path'], str(paths[1]))
            self.assertEqual(json.loads(report.read_text())['winner'], selected['winner'])
            self.assertEqual([path.read_bytes() for path in paths], original)
            for call in evaluate.call_args_list:
                self.assertEqual(call.kwargs['selection'], 'suite')
                self.assertEqual(call.kwargs['seeds'], [17, 18])
                self.assertEqual(call.kwargs['opponent_sha256'], hashlib.sha256(b'frozen').hexdigest())
            self.assertEqual(save.call_args.args, (output, 'b'))
            self.assertEqual(save.call_args.kwargs['metadata']['source_checkpoint'], str(paths[1]))
            self.assertTrue(save.call_args.kwargs['metadata']['search']['proven_win_priority'])

    def test_protects_inputs_existing_outputs_and_changed_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate, report = root / 'candidate.eqx', root / 'report.json'
            candidate.write_bytes(b'initial')
            with self.assertRaises(ValueError):
                select_checkpoints([candidate], report_path=candidate)
            report.write_text('retain')
            with self.assertRaises(FileExistsError):
                select_checkpoints([candidate], report_path=report)
            self.assertEqual(report.read_text(), 'retain')
            def changed(*args, **kwargs):
                candidate.write_bytes(b'changed')
                return {}
            with patch('connect4_jax.checkpoint.load_checkpoint', return_value=('model', {})), \
                 patch('connect4_jax.evaluation.evaluate', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed'):
                select_checkpoints([candidate], report_path=root / 'new.json')
            self.assertFalse((root / 'new.json').exists())
