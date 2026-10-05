"""Relative positional advantage for the GUI, not calibrated win probability."""
import math
from pathlib import Path

import torch

from agent import DuelingConnect4Net
from game import (AI_PIECE, PLAYER_PIECE, get_candidate_moves,
                  get_immediate_winning_move, get_state_tensor, get_valid_locations,
                  score_position, winning_move)


class PositionEvaluator:
    """Return red/yellow advantage shares and the evaluation source.

    DQN inputs represent the side to move. Evaluating both perspectives on the
    same board incorrectly gives both players the next move. Shares sum to one
    for drawing the bar; they are not empirical win probabilities.
    """
    def __init__(self, model_path=None, algorithm="dqn"):
        if algorithm not in ("dqn", "alphazero"):
            raise ValueError("Unknown evaluation algorithm")
        self.algorithm = algorithm
        self.equinox_agent = None
        eqx_path = "model/connect4_alphazero.eqx"
        model_path = model_path or ((eqx_path if Path(eqx_path).exists() else "model/connect4_alphazero.pth") if algorithm == "alphazero"
                                    else "model/connect4_model_selfplay.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if Path(model_path).exists():
            try:
                if algorithm == "alphazero":
                    if str(model_path).endswith(".eqx"):
                        from connect4_jax.agent import AlphaZeroAgent
                        self.equinox_agent = AlphaZeroAgent.from_checkpoint(model_path)
                        config = self.equinox_agent.model.config
                        if (config.rows, config.cols, config.connect) != (6, 7, 4):
                            self.equinox_agent = None
                            raise ValueError("GUI evaluation requires a standard Connect 4 checkpoint")
                        return
                    from alphazero import load_checkpoint
                    model, _ = load_checkpoint(model_path, self.device)
                    if (model.config.rows, model.config.cols, model.config.connect) != (6, 7, 4):
                        raise ValueError("GUI evaluation requires a standard Connect 4 checkpoint")
                else:
                    model = DuelingConnect4Net().to(self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                model.eval()
                self.model = model
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[PositionEvaluator] Using heuristic: {exc}")

    def evaluate(self, board, current_turn_piece=None):
        if winning_move(board, PLAYER_PIECE):
            return 1.0, 0.0, "Terminal"
        if winning_move(board, AI_PIECE):
            return 0.0, 1.0, "Terminal"
        if not get_valid_locations(board):
            return 0.5, 0.5, "Draw"
        if current_turn_piece in (PLAYER_PIECE, AI_PIECE):
            if self.equinox_agent is not None:
                _, value = self.equinox_agent.predict(board, current_turn_piece)
                if math.isfinite(value) and -1.0 <= value <= 1.0:
                    red_value = value if current_turn_piece == PLAYER_PIECE else -value
                    red = max(0.01, min(0.99, (1.0 + red_value) / 2.0))
                    return red, 1.0 - red, "AlphaZero value (Equinox)"
            if self.algorithm == "alphazero" and self.model is not None:
                from alphazero import Position
                position = Position.from_board(board, current_turn_piece, self.model.config)
                state = torch.from_numpy(position.encode()).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, estimate = self.model(state)
                value = float(estimate.item())
                red_value = value if current_turn_piece == PLAYER_PIECE else -value
                red = max(0.01, min(0.99, (1.0 + red_value) / 2.0))
                return red, 1.0 - red, "AlphaZero value"
            if get_immediate_winning_move(board, current_turn_piece) is not None:
                red = 0.999 if current_turn_piece == PLAYER_PIECE else 0.001
                return red, 1.0 - red, "Win next move"
            if self.model is not None:
                candidates = get_candidate_moves(board, current_turn_piece, 3 - current_turn_piece)
                state = get_state_tensor(board, current_turn_piece, 3 - current_turn_piece).to(self.device)
                with torch.no_grad():
                    q = self.model(state)[0].cpu().tolist()
                value = max(q[c] for c in candidates)
                # Old or unstable checkpoints can exceed the feasible return
                # range. Do not turn that into misleading near-certain odds.
                if math.isfinite(value) and -1.0 <= value <= 1.0:
                    red_value = value if current_turn_piece == PLAYER_PIECE else -value
                    red = max(0.01, min(0.99, (1.0 + red_value) / 2.0))
                    return red, 1.0 - red, "DQN estimate"
        difference = (score_position(board, PLAYER_PIECE, AI_PIECE)
                      - score_position(board, AI_PIECE, PLAYER_PIECE))
        red = 0.5 * (1.0 + math.tanh(difference / 28.0))
        red = max(0.01, min(0.99, red))
        return red, 1.0 - red, "Heuristic"

    def get_win_rates(self, board, current_turn_piece=None):
        """Compatibility alias; returned shares represent positional advantage."""
        return self.evaluate(board, current_turn_piece)


WinRateEvaluator = PositionEvaluator
