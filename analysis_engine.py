"""
AlphaGo-Style Connect 4 Analysis Engine.
Extracts real-time win rates, candidate move expected values (Q-values),
MCTS visit distributions, prior policies, principal variations (PV),
and tactical threat awareness without re-training models.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from game import (
    AI_PIECE,
    COL_COUNT,
    PLAYER_PIECE,
    ROW_COUNT,
    drop_piece,
    get_candidate_moves,
    get_immediate_winning_move,
    get_next_open_row,
    get_state_tensor,
    get_valid_locations,
    is_terminal_node,
    is_valid_location,
    score_position,
    winning_move,
)


@dataclass
class CandidateMove:
    col: int
    win_rate: float          # [0.0, 100.0] from current player's perspective
    visits: int              # MCTS visit count
    visit_share: float       # [0.0, 1.0] percentage of total visits
    prior: float             # [0.0, 1.0] raw policy prior before search
    q_value: float           # [-1.0, 1.0] expected utility
    is_best: bool = False
    rank: int = 1
    pv: List[int] = field(default_factory=list)  # Predicted continuation if this move is chosen


@dataclass
class TacticalInfo:
    immediate_wins: List[int] = field(default_factory=list)
    opponent_threats: List[int] = field(default_factory=list)
    forbidden_moves: List[int] = field(default_factory=list)  # Col where piece allows opponent to win on row above


@dataclass
class AnalysisResult:
    to_play: int             # 1 = Red, 2 = Yellow
    root_value: float        # [-1.0, 1.0] from side-to-move perspective
    win_rate_side: float     # [0.0, 100.0] side-to-move win rate
    win_rate_red: float      # [0.0, 100.0] Red win rate
    win_rate_yellow: float   # [0.0, 100.0] Yellow win rate
    candidates: List[CandidateMove] = field(default_factory=list)
    best_move: Optional[int] = None
    principal_variation: List[int] = field(default_factory=list)
    tactical: TacticalInfo = field(default_factory=TacticalInfo)
    eval_source: str = "AlphaZero"
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_tactical_threats(board: np.ndarray, to_play: int) -> TacticalInfo:
    """Analyze immediate winning, blocking, and forbidden blunder moves."""
    opp = 3 - to_play
    valid_cols = get_valid_locations(board)
    immediate_wins = []
    opponent_threats = []
    forbidden_moves = []

    # 1. Immediate win for to_play
    for c in valid_cols:
        r = get_next_open_row(board, c)
        b_copy = board.copy()
        drop_piece(b_copy, r, c, to_play)
        if winning_move(b_copy, to_play):
            immediate_wins.append(c)

    # 2. Immediate opponent win (threat to block)
    for c in valid_cols:
        r = get_next_open_row(board, c)
        b_copy = board.copy()
        drop_piece(b_copy, r, c, opp)
        if winning_move(b_copy, opp):
            opponent_threats.append(c)

    # 3. Forbidden move: placing in col c gives opponent winning move immediately on r+1
    for c in valid_cols:
        r = get_next_open_row(board, c)
        if r + 1 < ROW_COUNT:
            b_copy = board.copy()
            drop_piece(b_copy, r, c, to_play)
            b_copy_opp = b_copy.copy()
            drop_piece(b_copy_opp, r + 1, c, opp)
            if winning_move(b_copy_opp, opp) and c not in immediate_wins:
                forbidden_moves.append(c)

    return TacticalInfo(
        immediate_wins=immediate_wins,
        opponent_threats=opponent_threats,
        forbidden_moves=forbidden_moves,
    )


def classify_move_quality(delta_win_rate_pct: float) -> Tuple[str, str]:
    """
    Classify human or AI move quality based on win rate delta drop.
    delta_win_rate_pct: win_rate(best_move) - win_rate(played_move)
    Returns: (label, badge_color_hex)
    """
    if delta_win_rate_pct <= 1.0:
        return "Best Move", "#10B981"      # Emerald green
    elif delta_win_rate_pct <= 3.5:
        return "Excellent", "#34D399"      # Light green
    elif delta_win_rate_pct <= 7.0:
        return "Good", "#60A5FA"           # Blue
    elif delta_win_rate_pct <= 12.0:
        return "Inaccuracy", "#FBBF24"     # Amber / Yellow
    elif delta_win_rate_pct <= 22.0:
        return "Mistake", "#F97316"        # Orange
    else:
        return "Blunder", "#EF4444"        # Red


class Connect4AnalysisEngine:
    """
    Comprehensive AlphaGo-style analysis engine supporting:
    - AlphaZero (Equinox / JAX & PyTorch)
    - Deep Q-Network (Dueling DQN)
    - Minimax
    """

    def __init__(self, default_simulations: int = 128):
        self.default_simulations = default_simulations
        self._equinox_agent = None
        self._torch_az_agent = None
        self._dqn_net = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()

    def _init_models(self):
        # 1. Try loading Equinox AlphaZero checkpoint
        eqx_candidates = [
            "model/experiments/larger_refined/latest_best.eqx",
            "model/connect4_alphazero.eqx",
            "model/experiments/larger/latest_best.eqx",
            "model/experiments/improved/latest_best.eqx",
        ]
        for path in eqx_candidates:
            if Path(path).exists():
                try:
                    from connect4_jax.agent import AlphaZeroAgent as EqxAgent
                    self._equinox_agent = EqxAgent.from_checkpoint(path, simulations=self.default_simulations)
                    self._equinox_agent.warmup()
                    print(f"[AnalysisEngine] Loaded Equinox AlphaZero: {path}")
                    break
                except Exception as e:
                    print(f"[AnalysisEngine] Could not load Equinox model {path}: {e}")

        # 2. Try loading DQN checkpoint
        dqn_path = Path("model/connect4_model_selfplay.pth")
        if dqn_path.exists():
            try:
                from agent import DuelingConnect4Net
                net = DuelingConnect4Net().to(self._device)
                net.load_state_dict(torch.load(str(dqn_path), map_location=self._device, weights_only=True))
                net.eval()
                self._dqn_net = net
                print(f"[AnalysisEngine] Loaded DQN: {dqn_path}")
            except Exception as e:
                print(f"[AnalysisEngine] Could not load DQN model: {e}")

    def analyze(
        self,
        board: np.ndarray,
        to_play: int = 1,
        engine_type: str = "alphazero",
        simulations: Optional[int] = None,
    ) -> AnalysisResult:
        """
        Analyze current board state from to_play's perspective.
        Returns comprehensive AlphaGo-style evaluation metrics.
        """
        start_time = time.perf_counter()
        sims = simulations or self.default_simulations
        tactical = detect_tactical_threats(board, to_play)
        valid_cols = get_valid_locations(board)

        # Check terminal state
        if winning_move(board, 1):
            return AnalysisResult(
                to_play=to_play,
                root_value=1.0 if to_play == 1 else -1.0,
                win_rate_side=100.0 if to_play == 1 else 0.0,
                win_rate_red=100.0,
                win_rate_yellow=0.0,
                best_move=None,
                eval_source="Terminal (Red Won)",
                tactical=tactical,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        if winning_move(board, 2):
            return AnalysisResult(
                to_play=to_play,
                root_value=-1.0 if to_play == 1 else 1.0,
                win_rate_side=0.0 if to_play == 1 else 100.0,
                win_rate_red=0.0,
                win_rate_yellow=100.0,
                best_move=None,
                eval_source="Terminal (Yellow Won)",
                tactical=tactical,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        if not valid_cols:
            return AnalysisResult(
                to_play=to_play,
                root_value=0.0,
                win_rate_side=50.0,
                win_rate_red=50.0,
                win_rate_yellow=50.0,
                best_move=None,
                eval_source="Terminal (Draw)",
                tactical=tactical,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        if engine_type == "alphazero" and self._equinox_agent is not None:
            res = self._analyze_equinox(board, to_play, sims, tactical)
        elif engine_type == "dqn" and self._dqn_net is not None:
            res = self._analyze_dqn(board, to_play, tactical)
        else:
            res = self._analyze_heuristic(board, to_play, tactical)

        res.latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        return res

    def _analyze_equinox(
        self,
        board: np.ndarray,
        to_play: int,
        sims: int,
        tactical: TacticalInfo,
    ) -> AnalysisResult:
        import jax
        from connect4_jax.environment import from_board
        from connect4_jax.search import SearchConfig, search

        agent = self._equinox_agent
        config = agent.model.config
        state = from_board(board, to_play, config)
        states = jax.tree.map(lambda x: x[None], state)
        search_settings = SearchConfig(simulations=sims, c_puct=1.5, policy="puct")

        res = search(agent.model, states, jax.random.PRNGKey(0), search_settings)
        single_tree = jax.tree.map(lambda x: x[0], res.search_tree)

        # 1. Root search metrics
        raw_root_val = float(single_tree.node_values[0])
        root_visits = np.asarray(single_tree.children_visits[0])
        root_qvalues = np.asarray(single_tree.qvalues(0))
        root_prior_logits = np.asarray(single_tree.children_prior_logits[0])

        # Prior probabilities
        legal_mask = np.zeros(COL_COUNT, dtype=bool)
        for c in get_valid_locations(board):
            legal_mask[c] = True

        valid_logits = np.where(legal_mask, root_prior_logits, -1e9)
        shifted = valid_logits - np.max(valid_logits)
        exp_logits = np.exp(shifted)
        priors = exp_logits / (np.sum(exp_logits) + 1e-12)

        total_visits = float(root_visits.sum())
        candidates: List[CandidateMove] = []

        # 2. PV extraction helper for a node index
        def extract_pv_from_node(tree, start_node: int, max_depth: int = 5) -> List[int]:
            pv_line = []
            curr = start_node
            for _ in range(max_depth):
                visits = np.asarray(tree.children_visits[curr])
                if visits.max() <= 0:
                    break
                next_act = int(np.argmax(visits))
                pv_line.append(next_act)
                child_idx = int(tree.children_index[curr, next_act])
                if child_idx == tree.UNVISITED:
                    break
                curr = child_idx
            return pv_line

        # 3. Assemble candidate moves
        for col in get_valid_locations(board):
            v = int(root_visits[col])
            share = (v / total_visits) if total_visits > 0 else 0.0
            q = float(root_qvalues[col])
            # Value in [-1, 1] maps to win percentage [0, 100]
            win_pct = max(0.1, min(99.9, ((q + 1.0) / 2.0) * 100.0))
            prior = float(priors[col])

            # Extract PV line starting from this child
            child_node = int(single_tree.children_index[0, col])
            continuation = [col]
            if child_node != single_tree.UNVISITED:
                continuation.extend(extract_pv_from_node(single_tree, child_node, max_depth=5))

            candidates.append(
                CandidateMove(
                    col=col,
                    win_rate=round(win_pct, 1),
                    visits=v,
                    visit_share=round(share, 3),
                    prior=round(prior, 3),
                    q_value=round(q, 3),
                    pv=continuation,
                )
            )

        # Sort candidates primarily by visit count, then Q-value
        candidates.sort(key=lambda c: (c.visits, c.q_value), reverse=True)
        for rank, c in enumerate(candidates, start=1):
            c.rank = rank
            c.is_best = (rank == 1)

        best_move = candidates[0].col if candidates else None
        main_pv = candidates[0].pv if candidates else []

        # Root win rate
        root_val_clipped = max(-1.0, min(1.0, raw_root_val))
        win_rate_side = round(((root_val_clipped + 1.0) / 2.0) * 100.0, 1)
        win_rate_red = win_rate_side if to_play == 1 else round(100.0 - win_rate_side, 1)
        win_rate_yellow = round(100.0 - win_rate_red, 1)

        return AnalysisResult(
            to_play=to_play,
            root_value=round(root_val_clipped, 3),
            win_rate_side=win_rate_side,
            win_rate_red=win_rate_red,
            win_rate_yellow=win_rate_yellow,
            candidates=candidates,
            best_move=best_move,
            principal_variation=main_pv,
            tactical=tactical,
            eval_source=f"AlphaZero MCTS ({sims} sims)",
        )

    def _analyze_dqn(self, board: np.ndarray, to_play: int, tactical: TacticalInfo) -> AnalysisResult:
        opp = 3 - to_play
        state = get_state_tensor(board, to_play, opp).to(self._device)
        with torch.no_grad():
            q_scores = self._dqn_net(state)[0].cpu().numpy()

        legal_cols = get_valid_locations(board)
        candidates: List[CandidateMove] = []

        # Softmax for prior / pseudo-visits
        legal_q = np.array([q_scores[c] for c in legal_cols])
        exp_q = np.exp(legal_q - np.max(legal_q))
        probs = exp_q / exp_q.sum()

        for idx, col in enumerate(legal_cols):
            q = float(q_scores[col])
            win_pct = max(0.1, min(99.9, ((q + 1.0) / 2.0) * 100.0))
            p = float(probs[idx])
            candidates.append(
                CandidateMove(
                    col=col,
                    win_rate=round(win_pct, 1),
                    visits=int(p * 100),
                    visit_share=round(p, 3),
                    prior=round(p, 3),
                    q_value=round(q, 3),
                    pv=[col],
                )
            )

        candidates.sort(key=lambda c: c.q_value, reverse=True)
        for rank, c in enumerate(candidates, start=1):
            c.rank = rank
            c.is_best = (rank == 1)

        best_move = candidates[0].col if candidates else None
        root_val = candidates[0].q_value if candidates else 0.0
        win_rate_side = round(((root_val + 1.0) / 2.0) * 100.0, 1)
        win_rate_red = win_rate_side if to_play == 1 else round(100.0 - win_rate_side, 1)

        return AnalysisResult(
            to_play=to_play,
            root_value=round(root_val, 3),
            win_rate_side=win_rate_side,
            win_rate_red=win_rate_red,
            win_rate_yellow=round(100.0 - win_rate_red, 1),
            candidates=candidates,
            best_move=best_move,
            principal_variation=[best_move] if best_move is not None else [],
            tactical=tactical,
            eval_source="DQN Q-Network",
        )

    def _analyze_heuristic(self, board: np.ndarray, to_play: int, tactical: TacticalInfo) -> AnalysisResult:
        diff = score_position(board, to_play, 3 - to_play) - score_position(board, 3 - to_play, to_play)
        val = math.tanh(diff / 28.0)
        win_rate_side = round(((val + 1.0) / 2.0) * 100.0, 1)
        win_rate_red = win_rate_side if to_play == 1 else round(100.0 - win_rate_side, 1)

        candidates: List[CandidateMove] = []
        for col in get_valid_locations(board):
            r = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, r, col, to_play)
            sub_diff = score_position(b_copy, to_play, 3 - to_play) - score_position(b_copy, 3 - to_play, to_play)
            sub_val = math.tanh(sub_diff / 28.0)
            sub_win = round(((sub_val + 1.0) / 2.0) * 100.0, 1)
            candidates.append(
                CandidateMove(
                    col=col,
                    win_rate=sub_win,
                    visits=1,
                    visit_share=1.0 / len(get_valid_locations(board)),
                    prior=1.0 / len(get_valid_locations(board)),
                    q_value=round(sub_val, 3),
                    pv=[col],
                )
            )

        candidates.sort(key=lambda c: c.q_value, reverse=True)
        for rank, c in enumerate(candidates, start=1):
            c.rank = rank
            c.is_best = (rank == 1)

        return AnalysisResult(
            to_play=to_play,
            root_value=round(val, 3),
            win_rate_side=win_rate_side,
            win_rate_red=win_rate_red,
            win_rate_yellow=round(100.0 - win_rate_red, 1),
            candidates=candidates,
            best_move=candidates[0].col if candidates else None,
            principal_variation=[candidates[0].col] if candidates else [],
            tactical=tactical,
            eval_source="Positional Heuristic",
        )
