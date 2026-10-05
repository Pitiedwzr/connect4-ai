import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from agent import DQNAgent, DuelingConnect4Net, ReplayMemory, policy_candidates_from_state
from benchmark import evaluate_network, network_move
from evaluator import PositionEvaluator
from game import (create_board, drop_piece, get_candidate_moves, get_next_open_row,
                  get_state_tensor, get_valid_locations, get_winning_moves,
                  is_suicide_move, winning_move)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def board_from_history(history):
    board = create_board()
    for ply, col in enumerate(history):
        drop_piece(board, get_next_open_row(board, col), col, 1 + ply % 2)
    return board


def brute_wins(board, piece):
    wins = []
    for col in get_valid_locations(board):
        after = board.copy()
        drop_piece(after, get_next_open_row(after, col), col, piece)
        if winning_move(after, piece):
            wins.append(col)
    return wins


class FixedQNet(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.calls = 0

    def forward(self, state):
        self.calls += 1
        return self.q.unsqueeze(0).expand(state.shape[0], -1)


class TacticalPolicyTests(unittest.TestCase):
    def test_center_opening_even_when_network_prefers_edge(self):
        net = FixedQNet([100, 0, 0, -10, 0, 0, 0])
        self.assertEqual(network_move(net, create_board(), 1, 2), 3)
        self.assertEqual(network_move(net, create_board(), 1, 2, raw=True), 0)

    def test_win_before_block(self):
        board = board_from_history([0, 6, 1, 6, 2, 6])
        self.assertEqual(get_candidate_moves(board, 1, 2), [3])

    def test_forced_block(self):
        board = board_from_history([0, 3, 0, 3, 1, 3])
        self.assertEqual(get_candidate_moves(board, 1, 2), [3])

    def test_original_gui_support_trap(self):
        board = board_from_history([4, 3, 3, 0, 1, 6, 3, 3, 6, 4, 5, 0, 6, 6])
        self.assertFalse(get_winning_moves(board, 1))
        self.assertFalse(get_winning_moves(board, 2))
        self.assertTrue(is_suicide_move(board, 5, 1, 2))
        self.assertNotIn(5, get_candidate_moves(board, 1, 2))
        net = FixedQNet([0, 0, 0, 0, 0, 100, 0])
        self.assertNotEqual(network_move(net, board, 1, 2), 5)

    def test_fork_trap_without_immediate_losing_reply(self):
        board = board_from_history([4, 3, 6, 2, 2, 2, 1, 6, 5, 5, 4, 5, 2])
        self.assertFalse(is_suicide_move(board, 0, 2, 1))
        self.assertEqual(set(get_candidate_moves(board, 2, 1)), {1, 4, 5})
        after = board.copy()
        drop_piece(after, get_next_open_row(after, 0), 0, 2)
        forks = []
        for reply in get_valid_locations(after):
            next_board = after.copy()
            drop_piece(next_board, get_next_open_row(next_board, reply), reply, 1)
            if not brute_wins(next_board, 2) and len(brute_wins(next_board, 1)) >= 2:
                forks.append(reply)
        self.assertTrue(forks, "Excluded move must have an independently verified fork reply")

    def test_unavoidable_loss_still_returns_legal_moves(self):
        board = board_from_history([6, 1, 6, 2, 5, 3])
        self.assertEqual(set(brute_wins(board, 2)), {0, 4})
        self.assertEqual(set(get_candidate_moves(board, 1, 2)), set(get_valid_locations(board)))

    def test_draw_has_no_candidates(self):
        board = np.array([[1, 1, 2, 2, 1, 1, 2], [2, 2, 1, 1, 2, 2, 1]] * 3)
        self.assertFalse(winning_move(board, 1))
        self.assertFalse(winning_move(board, 2))
        self.assertEqual(get_candidate_moves(board, 1, 2), [])

    def test_fast_checks_symmetry_and_board_preservation(self):
        rng = random.Random(42)
        for _ in range(25):
            board = create_board()
            for ply in range(20):
                before = board.copy()
                for piece in (1, 2):
                    self.assertEqual(set(get_winning_moves(board, piece)), set(brute_wins(board, piece)))
                    moves = get_candidate_moves(board, piece, 3 - piece)
                    mirrored = get_candidate_moves(np.fliplr(board), piece, 3 - piece)
                    self.assertEqual({6 - col for col in moves}, set(mirrored))
                    self.assertTrue(set(moves).issubset(get_valid_locations(board)))
                np.testing.assert_array_equal(before, board)
                col = rng.choice(get_valid_locations(board))
                drop_piece(board, get_next_open_row(board, col), col, 1 + ply % 2)
                if winning_move(board, 1 + ply % 2):
                    break


class LearningTests(unittest.TestCase):
    def test_soft_update_copies_bn_buffers_and_interpolates_weights(self):
        agent = DQNAgent(device="cpu")
        with torch.no_grad():
            agent.policy_net.conv1.weight.fill_(2.0)
            agent.target_net.conv1.weight.zero_()
            agent.policy_net.bn1.running_mean.fill_(3.0)
            agent.policy_net.bn1.running_var.fill_(4.0)
            agent.policy_net.bn1.num_batches_tracked.fill_(17)
        agent.update_target_network(0.25)
        self.assertTrue(torch.all(agent.target_net.conv1.weight == 0.5))
        self.assertTrue(torch.all(agent.target_net.bn1.running_mean == 3.0))
        self.assertTrue(torch.all(agent.target_net.bn1.running_var == 4.0))
        self.assertEqual(agent.target_net.bn1.num_batches_tracked.item(), 17)

    def test_replay_stores_and_mirrors_actual_policy_mask(self):
        board = board_from_history([4, 3, 3, 0, 1, 6, 3, 3, 6, 4, 5, 0, 6, 6])
        state = get_state_tensor(board, 1, 2)
        self.assertEqual(set(policy_candidates_from_state(state)), set(get_candidate_moves(board, 1, 2)))
        memory = ReplayMemory()
        memory.push_with_symmetry(state, 1, 0, state, False)
        original, mirrored = list(memory.memory)
        self.assertFalse(original[5][5])
        self.assertTrue(torch.equal(torch.flip(original[5], [0]), mirrored[5]))
        self.assertEqual(mirrored[1], 5)
        memory.push(state, 1, -1, state, True)
        self.assertFalse(memory.memory[-1][5].any())

    def make_fixed_agent(self, target_values):
        agent = DQNAgent(device="cpu")
        agent.policy_net = FixedQNet([0, 0, 0, 0.25, 0, 0, 100])
        agent.target_net = FixedQNet(target_values)
        agent.target_net.eval()
        agent.optimizer = torch.optim.SGD(agent.policy_net.parameters(), lr=0.0)
        return agent

    def test_bootstrap_cannot_select_disallowed_high_q_action(self):
        agent = self.make_fixed_agent([0, 0, 0, 0.75, 0, 0, -1])
        memory = ReplayMemory()
        state = get_state_tensor(create_board(), 1, 2)
        mask = torch.tensor([False, False, False, True, False, False, False])
        memory.push(state, 0, 0, state, False, mask)
        loss = agent.learn(memory, 1)
        self.assertAlmostEqual(loss, 0.5 * (agent.gamma * 0.75) ** 2, places=6)

    def test_terminal_samples_never_bootstrap_even_with_nan_target(self):
        agent = self.make_fixed_agent([math.nan] * 7)
        memory = ReplayMemory()
        state = get_state_tensor(create_board(), 1, 2)
        memory.push(state, 0, -1, state, True)
        self.assertAlmostEqual(agent.learn(memory, 1), 0.5)
        self.assertEqual(agent.target_net.calls, 0)

    def test_bootstrap_is_bounded_by_possible_returns(self):
        agent = self.make_fixed_agent([0, 0, 0, 50, 0, 0, 0])
        memory = ReplayMemory()
        state = get_state_tensor(create_board(), 1, 2)
        memory.push(state, 0, 0, state, False)
        self.assertAlmostEqual(agent.learn(memory, 1), 0.5 * agent.gamma ** 2, places=6)

    def test_cpu_learning_with_real_batchnorm_network(self):
        agent = DQNAgent(device="cpu")
        memory = ReplayMemory()
        before = agent.policy_net.conv1.weight.detach().clone()
        for history in ([3, 2], [2, 3], [3, 3, 2, 4], [4, 3]):
            state = get_state_tensor(board_from_history(history), 1, 2)
            memory.push_with_symmetry(state, 3, 0, state, False)
        loss = agent.learn(memory, 8)
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(torch.equal(before, agent.policy_net.conv1.weight))
        self.assertEqual(agent.policy_net.bn1.num_batches_tracked.item(), 1)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_reproducible_and_preserves_training_state(self):
        net = DuelingConnect4Net()
        net.train()
        random.seed(13)
        rng_before = random.getstate()
        bn_before = net.bn1.running_mean.clone()
        first = evaluate_network(net, games=1, seed=123, depths=(2,))
        self.assertTrue(net.training)
        self.assertEqual(random.getstate(), rng_before)
        self.assertTrue(torch.equal(bn_before, net.bn1.running_mean))
        self.assertEqual(first, evaluate_network(net, games=1, seed=123, depths=(2,)))
        for opponent in first.values():
            for seat in opponent.values():
                self.assertEqual(sum(seat.values()), 1)

    def test_evaluator_uses_only_side_to_move_and_rejects_unbounded_q(self):
        evaluator = PositionEvaluator(model_path="missing-test-checkpoint.pth")
        evaluator.model = FixedQNet([0.5] * 7)
        red, yellow, source = evaluator.evaluate(create_board(), 2)
        self.assertEqual((red, yellow, source), (0.25, 0.75, "DQN estimate"))
        self.assertEqual(evaluator.model.calls, 1)
        evaluator.model = FixedQNet([2.5] * 7)
        self.assertEqual(evaluator.evaluate(create_board(), 1), (0.5, 0.5, "Heuristic"))

    def test_dqn_loader_fails_instead_of_using_random_weights(self):
        from ai_player import DQNPlayer
        with self.assertRaises(FileNotFoundError):
            DQNPlayer(model_path="missing-test-checkpoint.pth")

    def test_accelerate_cpu_training_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory(prefix="connect4-training-") as temp:
            output = Path(temp) / "smoke.pth"
            command = [sys.executable, str(ROOT / "train.py"), "--cpu", "--episodes", "4",
                       "--batch-size", "8", "--warmup-steps", "8", "--minimax-max-depth", "2",
                       "--snapshot-every", "1", "--pretrain-positions", "8", "--pretrain-depth", "2",
                       "--eval-every", "0", "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("device=cpu", result.stdout)
            net = DuelingConnect4Net()
            net.load_state_dict(torch.load(output, map_location="cpu", weights_only=True))
            self.assertGreater(net.bn1.num_batches_tracked.item(), 1)
            self.assertEqual(network_move(net.eval(), create_board(), 1, 2), 3)


if __name__ == "__main__":
    unittest.main()
