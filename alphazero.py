"""Policy/value learning and PUCT for gravity-based Connect-N games.

Only legal moves and exact terminal outcomes enter the search. There are no
opening rules, tactical shields, heuristic scores, or minimax teacher labels.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import threading

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "model/connect4_alphazero.pth"


@dataclass(frozen=True)
class AlphaZeroConfig:
    rows: int = 6
    cols: int = 7
    connect: int = 4
    channels: int = 64
    blocks: int = 4

    def __post_init__(self):
        if min(self.rows, self.cols, self.channels) < 1 or self.blocks < 0:
            raise ValueError("Board dimensions/channels must be positive; blocks must be nonnegative")
        if self.connect < 2 or self.connect > max(self.rows, self.cols):
            raise ValueError("connect must be between 2 and the longest board dimension")


def has_connect(bits, config):
    # Python integers also support boards exceeding 64 bits. Sentinel cells
    # prevent runs wrapping from the top of one column to the next column.
    stride = config.rows + 1
    for shift in (1, stride, stride - 1, stride + 1):
        run = bits
        for offset in range(1, config.connect):
            run &= bits >> (shift * offset)
        if run:
            return True
    return False


@dataclass(frozen=True)
class Position:
    config: AlphaZeroConfig
    first: int
    second: int
    heights: tuple
    to_play: int = 1
    winner: int = 0

    @classmethod
    def empty(cls, config=None):
        config = config or AlphaZeroConfig()
        return cls(config, 0, 0, (0,) * config.cols)

    @classmethod
    def from_board(cls, board, to_play, config=None):
        config = config or AlphaZeroConfig(rows=board.shape[0], cols=board.shape[1])
        board = np.asarray(board)
        if board.shape != (config.rows, config.cols) or to_play not in (1, 2):
            raise ValueError("Board shape or side to move does not match the game configuration")
        if not np.isin(board, (0, 1, 2)).all():
            raise ValueError("Board cells must be 0, 1 or 2")
        heights = tuple(int(n) for n in np.count_nonzero(board, axis=0))
        first = second = 0
        for col, height in enumerate(heights):
            if np.any(board[:height, col] == 0) or np.any(board[height:, col] != 0):
                raise ValueError("Pieces must obey gravity")
            for row in range(height):
                bit = 1 << (col * (config.rows + 1) + row)
                if board[row, col] == 1:
                    first |= bit
                else:
                    second |= bit
        wins = (has_connect(first, config), has_connect(second, config))
        if all(wins):
            raise ValueError("Both players cannot have winning lines")
        winner = 1 if wins[0] else 2 if wins[1] else 0
        return cls(config, first, second, heights, to_play, winner)

    @property
    def ply(self):
        return sum(self.heights)

    def terminal_value(self):
        """Outcome from the side-to-move perspective, or None if unfinished."""
        if self.winner:
            return 1.0 if self.winner == self.to_play else -1.0
        return 0.0 if self.ply == self.config.rows * self.config.cols else None

    def legal_moves(self):
        if self.terminal_value() is not None:
            return []
        return [col for col, height in enumerate(self.heights) if height < self.config.rows]

    def play(self, col):
        if col not in self.legal_moves():
            raise ValueError(f"Illegal move: {col}")
        bit = 1 << (col * (self.config.rows + 1) + self.heights[col])
        first = self.first | bit if self.to_play == 1 else self.first
        second = self.second | bit if self.to_play == 2 else self.second
        heights = list(self.heights)
        heights[col] += 1
        bits = first if self.to_play == 1 else second
        winner = self.to_play if has_connect(bits, self.config) else 0
        return Position(self.config, first, second, tuple(heights), 3 - self.to_play, winner)

    def encode(self):
        """Two planes: side-to-move pieces, then opponent pieces; bottom row first."""
        own, other = ((self.first, self.second) if self.to_play == 1
                      else (self.second, self.first))
        planes = np.zeros((2, self.config.rows, self.config.cols), dtype=np.float32)
        for col, height in enumerate(self.heights):
            for row in range(height):
                bit = 1 << (col * (self.config.rows + 1) + row)
                planes[0, row, col] = bool(own & bit)
                planes[1, row, col] = bool(other & bit)
        return planes

    def board(self):
        planes = self.encode()
        return (planes[0] * self.to_play + planes[1] * (3 - self.to_play)).astype(np.int64)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = math.gcd(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        residual = F.relu(self.norm1(self.conv1(x)))
        return F.relu(x + self.norm2(self.conv2(residual)))


class AlphaZeroNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or AlphaZeroConfig()
        c = self.config
        # GroupNorm makes single-leaf inference independent of batch statistics.
        self.trunk = nn.Sequential(
            nn.Conv2d(2, c.channels, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(8, c.channels), c.channels), nn.ReLU(),
            *(ResidualBlock(c.channels) for _ in range(c.blocks)))
        self.policy_conv = nn.Conv2d(c.channels, 2, 1)
        self.policy_fc = nn.Linear(2 * c.rows * c.cols, c.cols)
        self.value_conv = nn.Conv2d(c.channels, 1, 1)
        self.value_fc = nn.Linear(c.rows * c.cols, 64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, x):
        if x.shape[1:] != (2, self.config.rows, self.config.cols):
            raise ValueError("Input board dimensions do not match the network checkpoint")
        features = self.trunk(x)
        policy = self.policy_fc(F.relu(self.policy_conv(features)).flatten(1))
        value = F.relu(self.value_conv(features)).flatten(1)
        value = torch.tanh(self.value_out(F.relu(self.value_fc(value))))
        return policy, value


@dataclass
class Node:
    position: Position
    prior: float = 1.0
    base_prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)
    expanded: bool = False

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    policy: np.ndarray
    visits: np.ndarray
    value: float

    def action(self, rng=None, temperature=0.0):
        if not self.policy.any():
            return None
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and nonnegative")
        if temperature == 0:
            return int(self.visits.argmax())
        weights = np.zeros_like(self.policy, dtype=np.float64)
        visited = self.visits > 0
        logits = np.log(self.visits[visited]) / temperature
        weights[visited] = np.exp(logits - logits.max())
        weights /= weights.sum()
        return int((rng or np.random.default_rng()).choice(len(weights), p=weights))


class MCTS:
    def __init__(self, model, simulations=128, c_puct=1.5, rng=None,
                 dirichlet_alpha=0.3, noise_fraction=0.25, cache_size=10000):
        if simulations < 1 or c_puct <= 0 or dirichlet_alpha <= 0:
            raise ValueError("Simulations, c_puct and Dirichlet alpha must be positive")
        if not 0 <= noise_fraction <= 1 or cache_size < 1:
            raise ValueError("noise_fraction must be in [0,1] and cache_size positive")
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = rng or np.random.default_rng()
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.root = None

    def reset(self):
        self.root = None
        self.cache.clear()

    def advance(self, action):
        self.root = self.root.children.get(action) if self.root is not None else None

    def _root_for(self, position):
        if position.config != self.model.config:
            raise ValueError("Position configuration differs from network configuration")
        if self.root is not None and self.root.position != position:
            # A GUI player has observed an opponent reply since its last move.
            self.root = next((child for child in self.root.children.values()
                              if child.position == position), None)
        if self.root is None:
            self.root = Node(position)
        return self.root

    @staticmethod
    def _install_children(node, priors):
        node.children = {col: Node(node.position.play(col), float(priors[col]), float(priors[col]))
                         for col in node.position.legal_moves()}
        node.expanded = True

    @staticmethod
    def _expand_many(pairs):
        """Batch leaf inference across independent games sharing a frozen model."""
        if not pairs:
            return []
        model = pairs[0][0].model
        if any(search.model is not model for search, _ in pairs):
            raise ValueError("Batched search must share the same model instance")
        values = [None] * len(pairs)
        missing = []
        for index, (search, node) in enumerate(pairs):
            if node.position in search.cache:
                priors, value = search.cache[node.position]
                search.cache.move_to_end(node.position)
                MCTS._install_children(node, priors)
                values[index] = value
            else:
                missing.append(index)
        if missing:
            device = next(model.parameters()).device
            states = np.stack([pairs[index][1].position.encode() for index in missing])
            with torch.inference_mode():
                logits, predictions = model(torch.from_numpy(states).to(device))
            logits = logits.detach().float().cpu().numpy()
            predictions = predictions.detach().float().cpu().numpy().reshape(-1)
            if not np.isfinite(logits).all() or not np.isfinite(predictions).all():
                raise RuntimeError("Nonfinite policy/value predictions")
            for batch_index, index in enumerate(missing):
                search, node = pairs[index]
                legal = node.position.legal_moves()
                scores = logits[batch_index, legal].astype(np.float64)
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                priors = np.zeros(node.position.config.cols, dtype=np.float64)
                priors[legal] = probabilities
                value = float(predictions[batch_index])
                search.cache[node.position] = (priors, value)
                if len(search.cache) > search.cache_size:
                    search.cache.popitem(last=False)
                MCTS._install_children(node, priors)
                values[index] = value
        return values

    def _select_leaf(self, root):
        path = [root]
        node = root
        while node.expanded and node.children:
            scale = math.sqrt(max(1, node.visits))
            # Child values are from the opponent's perspective. Negate them
            # before comparing moves from this node's side-to-move perspective.
            node = max(node.children.values(), key=lambda child:
                       -child.value + self.c_puct * child.prior * scale / (1 + child.visits))
            path.append(node)
        return path

    @staticmethod
    def _backup(path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    @staticmethod
    def search_batch(searches, positions, add_noise=False):
        if len(searches) != len(positions) or len({id(s) for s in searches}) != len(searches):
            raise ValueError("Each position needs its own search instance")
        roots = [search._root_for(position) for search, position in zip(searches, positions)]
        pending = [(search, root) for search, root in zip(searches, roots)
                   if not root.expanded and root.position.terminal_value() is None]
        MCTS._expand_many(pending)
        for search, root in zip(searches, roots):
            children = list(root.children.values())
            if not children:
                continue
            noise = (search.rng.dirichlet([search.dirichlet_alpha] * len(children))
                     if add_noise else np.zeros(len(children)))
            fraction = search.noise_fraction if add_noise else 0.0
            for child, perturbation in zip(children, noise):
                child.prior = (1 - fraction) * child.base_prior + fraction * perturbation
        for simulation in range(max((s.simulations for s in searches), default=0)):
            paths, leaves = [], []
            for search, root in zip(searches, roots):
                if simulation >= search.simulations or root.position.terminal_value() is not None:
                    continue
                path = search._select_leaf(root)
                terminal = path[-1].position.terminal_value()
                if terminal is not None:
                    MCTS._backup(path, terminal)
                else:
                    paths.append(path)
                    leaves.append((search, path[-1]))
            for path, value in zip(paths, MCTS._expand_many(leaves)):
                MCTS._backup(path, value)
        results = []
        for root in roots:
            visits = np.zeros(root.position.config.cols, dtype=np.int64)
            for col, child in root.children.items():
                visits[col] = child.visits
            policy = visits / visits.sum() if visits.sum() else np.zeros(len(visits), dtype=np.float64)
            terminal = root.position.terminal_value()
            results.append(SearchResult(policy.astype(np.float32), visits,
                                        root.value if terminal is None else terminal))
        return results

    def search(self, position, add_noise=False):
        return self.search_batch([self], [position], add_noise)[0]


def policy_move(model, position):
    """Raw policy selection with legal masking, without tactical rules or MCTS."""
    legal = position.legal_moves()
    if not legal:
        return None
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits, _ = model(torch.from_numpy(position.encode()).unsqueeze(0).to(device))
    scores = logits[0].float().cpu().tolist()
    return max(legal, key=lambda col: scores[col])


def save_checkpoint(path, model, **training_state):
    payload = dict(format_version=1, algorithm="alphazero", config=asdict(model.config),
                   model_state=model.state_dict(), **training_state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replacing a complete sibling file preserves the last usable checkpoint
    # if training is interrupted during serialization.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("algorithm") != "alphazero" or payload.get("format_version") != 1:
        raise ValueError("Expected an AlphaZero checkpoint, not a DQN state dictionary")
    model = AlphaZeroNet(AlphaZeroConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


class AlphaZeroAgent:
    def __init__(self, model, simulations=128, c_puct=1.5):
        self.model = model.eval()
        self.search = MCTS(model, simulations, c_puct)
        self.last_result = None
        # GUI cancellation discards results but cannot stop a running worker.
        # Serialize access when a reset starts a second worker on this agent.
        self._lock = threading.Lock()

    def get_move(self, board, my_piece=1, opp_piece=2, raw=False):
        with self._lock:
            return self._get_move(board, my_piece, opp_piece, raw)

    def _get_move(self, board, my_piece, opp_piece, raw):
        if my_piece not in (1, 2) or opp_piece != 3 - my_piece:
            raise ValueError("AlphaZero uses pieces 1 and 2")
        position = Position.from_board(board, my_piece, self.model.config)
        if raw:
            return policy_move(self.model, position)
        self.last_result = self.search.search(position)
        action = self.last_result.action()
        if action is not None:
            self.search.advance(action)
        return action
