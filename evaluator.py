import os
import math
import numpy as np
import torch

from game import (
    winning_move,
    get_immediate_winning_move,
    get_valid_locations,
    score_position,
    get_state_tensor,
    PLAYER_PIECE,
    AI_PIECE,
)
from agent import DuelingConnect4Net


class WinRateEvaluator:
    """
    Evaluates real-time win rate for Player 1 (Red) and Player 2 (Yellow).
    Uses trained DQN model as an impartial judge if available,
    falling back seamlessly to heuristic evaluation.
    """

    def __init__(self, model_path="model/connect4_model.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.use_model = False

        if os.path.exists(self.model_path):
            try:
                net = DuelingConnect4Net().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                net.load_state_dict(checkpoint)
                net.eval()
                self.model = net
                self.use_model = True
            except Exception as e:
                print(f"[WinRateEvaluator] Failed to load DQN model: {e}. Falling back to heuristic.")
                self.use_model = False
        else:
            print(f"[WinRateEvaluator] Model file not found at {self.model_path}. Using heuristic.")
            self.use_model = False

    def evaluate(self, board, current_turn_piece=None):
        """
        Calculates win rate for Player 1 (Piece 1, Red) and Player 2 (Piece 2, Yellow).

        Returns:
            p1_win_rate (float): 0.0 to 1.0 (win probability for Player 1)
            p2_win_rate (float): 0.0 to 1.0 (win probability for Player 2)
            source_tag (str): "DQN 模型" or "启发式评估"
        """
        valid_moves = get_valid_locations(board)

        # 1. Terminal State Checks
        if winning_move(board, PLAYER_PIECE):
            return 1.0, 0.0, "终局判定"
        if winning_move(board, AI_PIECE):
            return 0.0, 1.0, "终局判定"
        if len(valid_moves) == 0:
            return 0.5, 0.5, "平局判定"

        # 2. Immediate Tactical Checks (1-ply win threat)
        if current_turn_piece is not None:
            if get_immediate_winning_move(board, current_turn_piece) is not None:
                if current_turn_piece == PLAYER_PIECE:
                    return 0.999, 0.001, "即时绝杀"
                else:
                    return 0.001, 0.999, "即时绝杀"

        # 3. DQN Model-based Evaluation
        if self.use_model and self.model is not None:
            try:
                # State from Player 1's perspective
                s1 = get_state_tensor(board, PLAYER_PIECE, AI_PIECE).to(self.device)
                # State from Player 2's perspective
                s2 = get_state_tensor(board, AI_PIECE, PLAYER_PIECE).to(self.device)

                with torch.no_grad():
                    q1 = self.model(s1)[0].cpu().numpy()
                    q2 = self.model(s2)[0].cpu().numpy()

                max_q1 = max(q1[c] for c in valid_moves)
                max_q2 = max(q2[c] for c in valid_moves)

                # Logistic sigmoid of Q-value difference with temperature tau
                tau = 0.45
                diff = max_q1 - max_q2
                p1_rate = 1.0 / (1.0 + math.exp(-diff / tau))
                p1_rate = max(0.01, min(0.99, p1_rate))
                return p1_rate, 1.0 - p1_rate, "DQN 模型"
            except Exception as e:
                # Fallback to heuristic on unexpected inference error
                pass

        # 4. Heuristic / Positional Score Fallback
        score_p1 = score_position(board, PLAYER_PIECE, AI_PIECE)
        score_p2 = score_position(board, AI_PIECE, PLAYER_PIECE)
        score_diff = score_p1 - score_p2

        # Temperature scaling for heuristic score
        temp = 14.0
        p1_rate = 1.0 / (1.0 + math.exp(-score_diff / temp))
        p1_rate = max(0.01, min(0.99, p1_rate))
        return p1_rate, 1.0 - p1_rate, "启发式评估"

    def get_win_rates(self, board, current_turn_piece=None):
        """Convenience alias for evaluate."""
        return self.evaluate(board, current_turn_piece)
