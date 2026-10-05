from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np
import torch
from torch import nn

from alphazero import (AlphaZeroAgent, AlphaZeroConfig, AlphaZeroNet, MCTS, Node,
                       Position, load_checkpoint, policy_move, save_checkpoint)
from benchmark import evaluate_network
from game import create_board, drop_piece, get_next_open_row, winning_move
from train_alphazero import SelfPlayReplay, policy_value_loss, self_play_games

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class FixedPolicyValue(nn.Module):
    def __init__(self, logits=None, value=0.0, config=None):
        super().__init__()
        self.config = config or AlphaZeroConfig()
        self.logits = nn.Parameter(torch.tensor(logits or [0.0] * self.config.cols))
        self.prediction = value
        self.batch_sizes = []

    def forward(self, states):
        self.batch_sizes.append(len(states))
        return self.logits.unsqueeze(0).expand(len(states), -1), torch.full((len(states), 1), self.prediction)


def fixture(history, to_play=1, config=None):
    config = config or AlphaZeroConfig()
    board = np.zeros((config.rows, config.cols), dtype=np.int64)
    for ply, col in enumerate(history):
        row = int(np.count_nonzero(board[:, col]))
        board[row, col] = 1 + ply % 2
    return Position.from_board(board, to_play, config)


class PositionTests(unittest.TestCase):
    def test_rules_match_existing_game_on_legal_positions(self):
        rng = random.Random(42)
        for _ in range(40):
            position = Position.empty()
            board = create_board()
            while position.terminal_value() is None:
                col = rng.choice(position.legal_moves())
                piece = position.to_play
                drop_piece(board, get_next_open_row(board, col), col, piece)
                position = position.play(col)
                np.testing.assert_array_equal(position.board(), board)
                self.assertEqual(position.winner == piece, winning_move(board, piece))
                if position.winner:
                    self.assertEqual(position.terminal_value(), -1.0)

    def test_larger_board_uses_python_bits_beyond_64(self):
        config = AlphaZeroConfig(rows=8, cols=9, channels=8, blocks=1)
        position = fixture([8, 0, 8, 0, 8, 1, 8], to_play=2, config=config)
        self.assertGreater(position.first.bit_length(), 64)
        self.assertEqual(position.winner, 1)
        self.assertEqual(position.terminal_value(), -1.0)

    def test_configurable_connect_and_terminal_legality(self):
        config = AlphaZeroConfig(rows=4, cols=5, connect=3, channels=8, blocks=1)
        position = fixture([0, 4, 1, 4, 2], to_play=2, config=config)
        self.assertEqual(position.winner, 1)
        self.assertEqual(position.legal_moves(), [])
        with self.assertRaises(ValueError):
            position.play(3)

    def test_draw_and_side_to_move_encoding(self):
        draw = np.array([[1, 1, 2, 2, 1, 1, 2], [2, 2, 1, 1, 2, 2, 1]] * 3)
        self.assertEqual(Position.from_board(draw, 1).terminal_value(), 0.0)
        first = fixture([3, 2], 1)
        second = Position.from_board(first.board(), 2)
        np.testing.assert_array_equal(first.encode()[0], second.encode()[1])
        np.testing.assert_array_equal(first.encode()[1], second.encode()[0])

    def test_rejects_invalid_gravity_and_shape(self):
        board = create_board()
        board[1, 0] = 1
        with self.assertRaises(ValueError):
            Position.from_board(board, 1)
        with self.assertRaises(ValueError):
            Position.from_board(create_board(), 1, AlphaZeroConfig(cols=8))


class NetworkTests(unittest.TestCase):
    def test_shapes_value_bounds_and_larger_boards(self):
        for config in (AlphaZeroConfig(channels=8, blocks=1),
                       AlphaZeroConfig(rows=8, cols=9, channels=8, blocks=1)):
            model = AlphaZeroNet(config)
            logits, values = model(torch.randn(2, 2, config.rows, config.cols))
            self.assertEqual(logits.shape, (2, config.cols))
            self.assertEqual(values.shape, (2, 1))
            self.assertTrue((values.abs() <= 1).all())
            (logits.mean() + values.mean()).backward()
            self.assertIsNotNone(model.trunk[0].weight.grad)
            with self.assertRaises(ValueError):
                model(torch.zeros(1, 2, config.rows + 1, config.cols))

    def test_group_norm_is_independent_of_leaf_batching(self):
        torch.manual_seed(17)
        model = AlphaZeroNet(AlphaZeroConfig(channels=8, blocks=1)).eval()
        first = torch.randn(1, 2, 6, 7)
        batch = torch.cat((first, torch.randn(3, 2, 6, 7)))
        with torch.no_grad():
            logits, values = model(first)
            batched_logits, batched_values = model(batch)
        torch.testing.assert_close(logits[0], batched_logits[0])
        torch.testing.assert_close(values[0], batched_values[0])


class SearchTests(unittest.TestCase):
    def test_no_forced_center_or_tactical_policy_in_root_expansion(self):
        model = FixedPolicyValue([10.0, 0, 0, -10, 0, 0, 0])
        search = MCTS(model, simulations=1)
        result = search.search(Position.empty(model.config))
        self.assertEqual(set(search.root.children), set(range(7)))
        self.assertEqual(result.action(), 0)

    def test_leaf_value_sign_and_alternating_backup(self):
        model = FixedPolicyValue(value=0.6)
        search = MCTS(model, simulations=1)
        self.assertAlmostEqual(search.search(Position.empty(model.config)).value, -0.6, places=6)
        path = [Node(Position.empty()), Node(fixture([0], 2)), Node(fixture([0, 1], 1))]
        MCTS._backup(path, 1.0)
        self.assertEqual([node.value for node in path], [1.0, -1.0, 1.0])

    def test_search_discovers_winning_move_using_terminal_rules(self):
        result = MCTS(FixedPolicyValue(), simulations=128).search(fixture([0, 6, 1, 6, 2, 5]))
        self.assertEqual(result.action(), 3)
        self.assertGreater(result.value, 0)

    def test_search_discovers_forced_block_without_a_shield(self):
        result = MCTS(FixedPolicyValue(), simulations=128).search(fixture([0, 3, 0, 3, 1, 3]))
        self.assertEqual(result.action(), 3)

    def test_full_column_mask_and_terminal_search(self):
        model = FixedPolicyValue([100.0, 0, 0, 0, 0, 0, 0])
        position = fixture([0] * 6)
        search = MCTS(model, simulations=16)
        result = search.search(position)
        self.assertEqual(result.policy[0], 0.0)
        self.assertNotIn(0, search.root.children)
        self.assertNotEqual(policy_move(model, position), 0)
        self.assertAlmostEqual(float(result.policy.sum()), 1.0)
        terminal = fixture([0, 6, 1, 6, 2, 5, 3], to_play=2)
        calls = len(model.batch_sizes)
        ended = search.search(terminal)
        self.assertIsNone(ended.action())
        self.assertEqual(ended.value, -1.0)
        self.assertEqual(len(model.batch_sizes), calls)

    def test_noise_is_seeded_and_does_not_accumulate_in_priors(self):
        model = FixedPolicyValue()
        first = MCTS(model, 8, rng=np.random.default_rng(123))
        second = MCTS(model, 8, rng=np.random.default_rng(123))
        first.search(Position.empty(), add_noise=True)
        second.search(Position.empty(), add_noise=True)
        priors = [child.prior for child in first.root.children.values()]
        np.testing.assert_allclose(priors, [child.prior for child in second.root.children.values()])
        self.assertGreater(np.std(priors), 0)
        first.search(Position.empty(), add_noise=False)
        np.testing.assert_allclose([child.prior for child in first.root.children.values()], [1 / 7] * 7)

    def test_batched_search_matches_individual_and_batches_inference(self):
        model = FixedPolicyValue()
        positions = [Position.empty(), fixture([3, 2])]
        searches = [MCTS(model, 16), MCTS(model, 16)]
        batched = MCTS.search_batch(searches, positions)
        self.assertIn(2, model.batch_sizes)
        for position, result in zip(positions, batched):
            separate = MCTS(model, 16).search(position)
            np.testing.assert_array_equal(result.visits, separate.visits)
            self.assertEqual(result.value, separate.value)

    def test_tree_reuse_and_cache_reset(self):
        search = MCTS(FixedPolicyValue(), simulations=16)
        position = Position.empty()
        result = search.search(position)
        child = search.root.children[result.action()]
        search.advance(result.action())
        self.assertIs(search._root_for(position.play(result.action())), child)
        self.assertTrue(search.cache)
        search.reset()
        self.assertFalse(search.cache)
        self.assertIsNone(search.root)

    def test_cached_agent_can_handle_overlapping_gui_requests(self):
        agent = AlphaZeroAgent(FixedPolicyValue(), simulations=128)
        boards = [fixture([0, 6, 1, 6, 2, 5]).board(), fixture([0, 3, 0, 3, 1, 3]).board()]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(agent.get_move, board, 1, 2) for board in boards]
            self.assertEqual([future.result(timeout=10) for future in futures], [3, 3])


class SelfPlayTrainingTests(unittest.TestCase):
    def test_larger_board_self_play_and_learning_step(self):
        torch.manual_seed(17)
        model = AlphaZeroNet(AlphaZeroConfig(rows=8, cols=9, channels=8, blocks=1))
        examples, metrics = self_play_games(model, games=2, simulations=2, rng=np.random.default_rng(17))
        replay = SelfPlayReplay()
        replay.add(examples)
        states, policies, outcomes = replay.sample(8, np.random.default_rng(17))
        self.assertEqual(states.shape, (8, 2, 8, 9))
        self.assertEqual(policies.shape, (8, 9))
        logits, values = model(states)
        loss, _, _ = policy_value_loss(logits, values, states, policies, outcomes)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_outcome_targets_are_correct_for_both_players(self):
        model = FixedPolicyValue()
        examples, metrics = self_play_games(model, games=2, simulations=4, rng=np.random.default_rng(42))
        self.assertEqual(metrics["first_wins"] + metrics["second_wins"] + metrics["draws"], 2)
        self.assertEqual(len(examples), metrics["positions"])
        game_start = 0
        for index, (state, policy, outcome) in enumerate(examples):
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
            self.assertTrue(np.isfinite(state).all())
            self.assertIn(outcome, (-1.0, 0.0, 1.0))
            occupied = int(state.sum())
            if occupied == 0:
                game_start = index
            else:
                expected = examples[game_start][2] * (-1 if occupied % 2 else 1)
                self.assertEqual(outcome, expected)
        self.assertIn(2, model.batch_sizes)

    def test_mirroring_and_replay_roundtrip(self):
        replay = SelfPlayReplay()
        state = fixture([3, 2]).encode()
        policy = np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        replay.add([(state, policy, -1.0)])
        first, second = replay.examples
        torch.testing.assert_close(torch.flip(first[0], [2]), second[0])
        self.assertEqual(second[1].argmax().item(), 5)
        self.assertEqual(second[2], -1.0)
        restored = SelfPlayReplay()
        restored.load_state_dict(replay.state_dict())
        torch.testing.assert_close(restored.examples[1][0], second[0])
        self.assertEqual(len(restored), 2)

    def test_masked_loss_is_finite_and_trains_both_heads(self):
        torch.manual_seed(17)
        model = AlphaZeroNet(AlphaZeroConfig(channels=8, blocks=1))
        state = torch.from_numpy(fixture([0] * 6, config=model.config).encode()).unsqueeze(0)
        policy = torch.tensor([[0.0, 0.5, 0.5, 0, 0, 0, 0]])
        logits, value = model(state)
        loss, policy_loss, value_loss = policy_value_loss(logits, value, state, policy, torch.tensor([1.0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.policy_fc.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.value_out.weight.grad.abs().sum().item(), 0)
        invalid = policy.clone()
        invalid[0, 0] = 0.1
        with self.assertRaises(ValueError):
            policy_value_loss(logits, value, state, invalid, torch.tensor([1.0]))

    def test_checkpoint_metadata_and_weights_only_reload(self):
        config = AlphaZeroConfig(rows=8, cols=9, channels=8, blocks=1)
        model = AlphaZeroNet(config)
        with tempfile.TemporaryDirectory(prefix="connect4-alpha-") as temp:
            path = Path(temp) / "alpha.pth"
            save_checkpoint(path, model, iteration=7)
            restored, payload = load_checkpoint(path)
            self.assertEqual(asdict(restored.config), asdict(config))
            self.assertEqual(payload["iteration"], 7)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, restored.state_dict()[name])
            old = Path(temp) / "dqn.pth"
            torch.save({"out.weight": torch.zeros(7, 1)}, old)
            with self.assertRaises(ValueError):
                load_checkpoint(old)

    def test_cpu_training_resume_and_benchmark_cli(self):
        with tempfile.TemporaryDirectory(prefix="connect4-alpha-training-") as temp:
            output = Path(temp) / "alpha.pth"
            log = Path(temp) / "training.jsonl"
            command = [sys.executable, str(ROOT / "train_alphazero.py"), "--cpu", "--iterations", "1",
                       "--games-per-iteration", "2", "--simulations", "4", "--channels", "8", "--blocks", "1",
                       "--batch-size", "8", "--warmup-positions", "8", "--updates-per-iteration", "2",
                       "--output", str(output), "--log", str(log)]
            trained = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=90)
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            model, payload = load_checkpoint(output)
            self.assertEqual(payload["iteration"], 1)
            self.assertEqual(payload["games_played"], 2)
            self.assertTrue(payload["replay"]["outcomes"].numel())
            previous = model.policy_fc.weight.detach().clone()
            resume_command = [sys.executable, str(ROOT / "train_alphazero.py"), "--cpu",
                              "--resume", str(output), "--iterations", "1", "--log", str(log)]
            resumed = subprocess.run(resume_command, cwd=ROOT,
                                     text=True, capture_output=True, timeout=90)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            model, payload = load_checkpoint(output)
            self.assertEqual(payload["iteration"], 2)
            self.assertEqual(payload["games_played"], 4)
            self.assertEqual(payload["training_args"]["simulations"], 4)
            self.assertEqual(payload["training_args"]["batch_size"], 8)
            self.assertEqual(payload["training_args"]["output"], str(output))
            self.assertEqual(payload["training_args"]["channels"], 8)
            self.assertFalse(torch.equal(previous, model.policy_fc.weight))
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 2)
            benchmark = [sys.executable, str(ROOT / "benchmark.py"), "--agent", "alphazero", "--cpu",
                         "--model", str(output), "--games", "1", "--depths", "2", "--simulations", "8"]
            for extra in ([], ["--raw"]):
                result = subprocess.run(benchmark + extra, cwd=ROOT, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["mode"], "raw" if extra else "mcts")
                self.assertEqual(report["agent"], "alphazero")
                self.assertIn("minimax_2", report["results"])

    def test_alpha_agent_works_in_existing_benchmark(self):
        model = AlphaZeroNet(AlphaZeroConfig(channels=8, blocks=1))
        player = AlphaZeroAgent(model, simulations=4)
        results = evaluate_network(model, games=1, depths=(), move_fn=player.get_move)
        self.assertIn("tactical_random", results)


if __name__ == "__main__":
    unittest.main()
