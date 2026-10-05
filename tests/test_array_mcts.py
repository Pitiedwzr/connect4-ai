import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from alphazero import AlphaZeroConfig, AlphaZeroNet, MCTS, Position
from array_mcts import ArrayMCTS
from train_alphazero import SelfPlayReplay, policy_value_loss, self_play_games

torch.set_num_threads(1)


class FixedNet(nn.Module):
    def __init__(self, config=None, value=0.0):
        super().__init__()
        self.config = config or AlphaZeroConfig()
        self.bias = nn.Parameter(torch.arange(self.config.cols, dtype=torch.float32) * 0.03)
        self.value = value

    def forward(self, states):
        return self.bias.expand(len(states), -1), torch.full((len(states), 1), self.value, device=states.device)


def position_after(moves, config=None):
    position = Position.empty(config)
    for col in moves:
        position = position.play(col)
    return position


class ArraySearchTests(unittest.TestCase):
    def test_visits_and_value_match_reference_with_nonzero_predictions(self):
        model = FixedNet(value=0.37)
        positions = [Position.empty(), position_after([3, 2, 4, 3]), position_after([0] * 6)]
        results = ArrayMCTS(model, positions, simulations=32).search()
        for position, result in zip(positions, results):
            reference = MCTS(model, simulations=32).search(position)
            np.testing.assert_array_equal(result.visits, reference.visits)
            self.assertAlmostEqual(result.value, reference.value, places=6)
            self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=6)
        self.assertEqual(results[2].policy[0], 0)

    def test_forced_win_and_block(self):
        positions = [position_after([0, 6, 1, 6, 2, 5]), position_after([0, 3, 0, 3, 1, 3])]
        results = ArrayMCTS(FixedNet(), positions, simulations=128).search()
        self.assertEqual([result.action() for result in results], [3, 3])

    def test_larger_boards_and_connect_three_rules_match_reference(self):
        rng = np.random.default_rng(9)
        for config in (AlphaZeroConfig(rows=8, cols=9), AlphaZeroConfig(rows=4, cols=5, connect=3)):
            position = Position.empty(config)
            model = FixedNet(config)
            search = ArrayMCTS(model, [position], simulations=2)
            while position.terminal_value() is None:
                result = search.search()[0]
                action = result.action(rng, 1.0)
                position = position.play(action)
                search.advance([action])
                root = int(search.roots[0])
                np.testing.assert_array_equal(search.boards[0, root].numpy(), position.encode())
                terminal = float(search.terminal[0, root])
                if position.terminal_value() is None:
                    self.assertTrue(np.isnan(terminal))
                else:
                    self.assertEqual(terminal, position.terminal_value())

    def test_terminal_rows_do_not_affect_active_game(self):
        terminal = position_after([0, 6, 1, 6, 2, 5, 3])
        model = FixedNet()
        results = ArrayMCTS(model, [terminal, Position.empty()], simulations=16).search(add_noise=True)
        self.assertIsNone(results[0].action())
        self.assertEqual(results[0].value, -1)
        separate = ArrayMCTS(model, [Position.empty()], simulations=16).search()[0]
        results = ArrayMCTS(model, [terminal, Position.empty()], simulations=16).search()
        np.testing.assert_array_equal(results[1].visits, separate.visits)

    def test_noise_is_repeatable_and_root_priors_are_restored(self):
        model = FixedNet()
        first = ArrayMCTS(model, [Position.empty()], simulations=8, rng=np.random.default_rng(21))
        second = ArrayMCTS(model, [Position.empty()], simulations=8, rng=np.random.default_rng(21))
        np.testing.assert_array_equal(first.search(True)[0].visits, second.search(True)[0].visits)
        torch.testing.assert_close(first.priors[0, 0], torch.softmax(model.bias.detach(), 0))

    def test_tree_reuse_preserves_visits_and_lazy_allocation(self):
        search = ArrayMCTS(FixedNet(), [Position.empty()], simulations=16)
        result = search.search()[0]
        self.assertLessEqual(int(search.counts[0]), 17)
        child = int(search.children[0, 0, result.action()])
        previous = int(search.visits[0, child])
        search.advance([result.action()])
        self.assertEqual(int(search.roots[0]), child)
        search.search()
        self.assertEqual(int(search.visits[0, child]), previous + 16)

    def test_memory_and_node_limits_fail_explicitly(self):
        model = FixedNet()
        with self.assertRaisesRegex(ValueError, "MiB"):
            ArrayMCTS(model, [Position.empty()], memory_mb=0.001)
        search = ArrayMCTS(model, [Position.empty()], simulations=4, max_nodes=5)
        search.search()
        with self.assertRaisesRegex(RuntimeError, "node limit"):
            search.search()
        with self.assertRaisesRegex(ValueError, "CUDA network"):
            ArrayMCTS(model, [Position.empty()], backend="cuda")

    def test_nonfinite_predictions_fail_without_invalid_tree_indices(self):
        model = FixedNet(value=float("nan"))
        with torch.no_grad():
            model.bias[0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            ArrayMCTS(model, [Position.empty()], simulations=4).search()

    def test_training_cli_resume_preserves_backend_and_reports_timings(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="connect4-array-") as temp:
            checkpoint = str(Path(temp) / "model.pth")
            log = Path(temp) / "log.jsonl"
            command = [sys.executable, "train_alphazero.py", "--cpu", "--iterations", "1",
                       "--search-backend", "cpu", "--profile-search", "--rows", "4", "--cols", "5",
                       "--connect", "3", "--channels", "8", "--blocks", "1", "--simulations", "2",
                       "--games-per-iteration", "2", "--batch-size", "8", "--warmup-positions", "8",
                       "--updates-per-iteration", "1", "--output", checkpoint, "--log", str(log)]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([sys.executable, "train_alphazero.py", "--cpu", "--resume",
                                     checkpoint, "--iterations", "1", "--log", str(log)],
                                    cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[-1]["games_played"], 4)
            for record in records:
                self.assertEqual(record["search_backend"], "cpu")
                self.assertGreater(record["collection_seconds"], 0)
                self.assertGreater(record["training_seconds"], 0)
                self.assertGreater(record["local_games_per_minute"], 0)
                self.assertGreater(record["search_profile"]["seconds"]["traversal"], 0)

    def test_self_play_targets_can_train_existing_network(self):
        model = AlphaZeroNet(AlphaZeroConfig(rows=4, cols=5, connect=3, channels=8, blocks=1))
        examples, metrics = self_play_games(model, games=2, simulations=4, search_backend="cpu",
                                            profile_search=True, rng=np.random.default_rng(19))
        self.assertEqual(sum(metrics[k] for k in ("first_wins", "second_wins", "draws")), 2)
        self.assertEqual(metrics["positions"], len(examples))
        self.assertGreater(metrics["search_profile"]["inference_calls"], 0)
        start = 0
        for index, (state, policy, outcome) in enumerate(examples):
            if state.sum() == 0:
                start = index
            self.assertEqual(outcome, examples[start][2] * (-1 if int(state.sum()) % 2 else 1))
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
            self.assertTrue((policy[state[:, -1].any(axis=0)] == 0).all())
        replay = SelfPlayReplay()
        replay.add(examples)
        states, policies, outcomes = replay.sample(8, np.random.default_rng(19))
        model.train()
        logits, values = model(states)
        loss, _, _ = policy_value_loss(logits, values, states, policies, outcomes)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"), "CUDA/Triton required")
    def test_cuda_kernels_match_tensor_reference_and_train(self):
        model = FixedNet().cuda()
        positions = [Position.empty(), position_after([0, 6, 1, 6, 2, 5]),
                     position_after([0, 3, 0, 3, 1, 3]), position_after([0] * 6)]
        cpu = ArrayMCTS(model, positions, simulations=128)
        cuda = ArrayMCTS(model, positions, simulations=128, backend="cuda")
        cuda_results = cuda.search()
        for left, right in zip(cpu.search(), cuda_results):
            # Floating-point tie decisions may differ between CPU and GPU.
            self.assertLessEqual(int(np.abs(left.visits - right.visits).max()), 2)
            self.assertAlmostEqual(left.value, right.value, delta=0.02)
        self.assertEqual([cuda_results[1].action(), cuda_results[2].action()], [3, 3])
        self.assertEqual(cuda_results[3].policy[0], 0)
        for search in (cpu, cuda):
            search.advance([3, 3, 3, 1])
        for left, right in zip(cpu.search(), cuda.search()):
            self.assertLessEqual(int(np.abs(left.visits - right.visits).max()), 2)
        net = AlphaZeroNet(AlphaZeroConfig(channels=8, blocks=1)).cuda()
        examples, _ = self_play_games(net, games=2, simulations=4, search_backend="cuda")
        replay = SelfPlayReplay()
        replay.add(examples)
        states, policies, outcomes = replay.sample(8, np.random.default_rng(19))
        net.train()
        logits, values = net(states.cuda())
        loss, _, _ = policy_value_loss(logits, values, states.cuda(), policies.cuda(), outcomes.cuda())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
