"""Batched PUCT with lazy nodes and device-resident tree statistics.

The CPU implementation is an executable reference for the tensor representation.
CUDA uses optional Triton kernels for traversal and backup, without a host read
inside a simulation. Networks/checkpoints remain ordinary PyTorch modules.
"""
from contextlib import contextmanager
import math
import time

import numpy as np
import torch

from alphazero import Position, SearchResult


class SearchProfile:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.seconds = {}
        self.inference_calls = 0
        self.inference_positions = 0

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start

    def report(self):
        return dict(seconds=self.seconds.copy(), inference_calls=self.inference_calls,
                    inference_positions=self.inference_positions,
                    mean_inference_batch=self.inference_positions / max(1, self.inference_calls))


def winning_lines(config):
    lines = []
    for row in range(config.rows):
        for col in range(config.cols):
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                end_row, end_col = row + dr * (config.connect - 1), col + dc * (config.connect - 1)
                if 0 <= end_row < config.rows and 0 <= end_col < config.cols:
                    lines.append([(row + dr * k) * config.cols + col + dc * k
                                  for k in range(config.connect)])
    return lines


class ArrayMCTS:
    def __init__(self, model, positions, simulations=128, c_puct=1.5,
                 dirichlet_alpha=0.3, noise_fraction=0.25, rng=None,
                 backend="cpu", max_nodes=0, memory_mb=512, profile=False):
        if not positions or simulations < 1 or c_puct <= 0 or dirichlet_alpha <= 0:
            raise ValueError("Positions, simulations, c_puct and alpha must be positive")
        if not 0 <= noise_fraction <= 1 or max_nodes < 0 or memory_mb <= 0:
            raise ValueError("Invalid noise fraction, node limit or memory budget")
        if backend not in ("cpu", "cuda"):
            raise ValueError("Array search backend must be cpu or cuda")
        self.model, self.config = model, model.config
        if any(p.config != self.config for p in positions):
            raise ValueError("Position configuration differs from the network")
        network_device = next(model.parameters()).device
        self.device = network_device if backend == "cuda" else torch.device("cpu")
        self.kernels = None
        if backend == "cuda":
            if network_device.type != "cuda":
                raise ValueError("CUDA search requires a CUDA network")
            try:
                import search_kernels
            except ImportError as exc:
                raise RuntimeError("CUDA search requires Triton (available with Linux CUDA PyTorch); "
                                   "use --search-backend python or cpu on other platforms") from exc
            self.kernels = search_kernels
        self.simulations, self.c_puct = simulations, c_puct
        self.alpha, self.noise_fraction = dirichlet_alpha, noise_fraction
        self.rng = rng or np.random.default_rng()
        self.positions = list(positions)
        self.games, self.cols = len(positions), self.config.cols
        self.cells = self.config.rows * self.cols
        # One lazy node per simulation, retained for every move of a game.
        self.capacity = max_nodes or 1 + self.cells * simulations
        if self.capacity < simulations + 1:
            raise ValueError("Node limit must accommodate a root and one search")
        # boards:uint8; children:int64; priors:float32; visits:int32;
        # sums/terminal:float32; expanded:bool. Include traversal scratch space.
        lines = winning_lines(self.config)
        self.memory_bytes = (self.games * (self.capacity * (2 * self.cells + 12 * self.cols + 13)
                                          + (self.cells + 1) * 8 + 65)
                             + len(lines) * self.config.connect * 8)
        if self.memory_bytes > memory_mb * 1024 ** 2:
            raise ValueError(f"Search arrays need approximately {self.memory_bytes / 1024**2:.1f} MiB; "
                             "increase --search-memory-mb or reduce concurrent games/simulations")
        d, b, n, c = self.device, self.games, self.capacity, self.cols
        self.boards = torch.zeros((b, n, 2, self.config.rows, c), dtype=torch.uint8, device=d)
        self.children = torch.full((b, n, c), -1, dtype=torch.int64, device=d)
        # Negative prior marks an illegal edge; legal edges may have zero mass.
        self.priors = torch.full((b, n, c), -1.0, device=d)
        self.visits = torch.zeros((b, n), dtype=torch.int32, device=d)
        self.sums = torch.zeros((b, n), device=d)
        self.terminal = torch.full((b, n), torch.nan, device=d)
        self.expanded = torch.zeros((b, n), dtype=torch.bool, device=d)
        self.roots = torch.zeros(b, dtype=torch.int64, device=d)
        self.counts = torch.ones(b, dtype=torch.int64, device=d)
        self.paths = torch.zeros((b, self.cells + 1), dtype=torch.int64, device=d)
        self.lengths = torch.ones(b, dtype=torch.int64, device=d)
        self.leaves = torch.zeros(b, dtype=torch.int64, device=d)
        self.parents = torch.zeros(b, dtype=torch.int64, device=d)
        self.actions = torch.zeros(b, dtype=torch.int64, device=d)
        self.created = torch.zeros(b, dtype=torch.bool, device=d)
        self.ids = torch.arange(b, device=d)
        self.lines = torch.tensor(lines, dtype=torch.int64, device=d)
        self.rounds = 0
        self.profile = SearchProfile(d, profile)
        self.finite = torch.ones((), dtype=torch.bool, device=d)
        for game, position in enumerate(positions):
            self.boards[game, 0] = torch.as_tensor(position.encode(), dtype=torch.uint8, device=d)
            outcome = position.terminal_value()
            if outcome is not None:
                self.terminal[game, 0] = outcome

    def _evaluate(self, nodes, mask):
        """Fixed batches avoid CUDA nonzero/host synchronization in simulations.

        Finished/cached rows are evaluated but masked out. This trades redundant
        small-network work for predictable inference shapes and no host roundtrip.
        """
        with self.profile.phase("encoding"):
            states = self.boards[self.ids, nodes].float()
        with self.profile.phase("inference"):
            logits, values = self.model(states.to(next(self.model.parameters()).device))
            logits = logits.float().to(self.device)
            values = values.reshape(-1).float().to(self.device)
        self.profile.inference_calls += 1
        self.profile.inference_positions += self.games
        with self.profile.phase("expansion"):
            self.finite &= logits.isfinite().all() & values.isfinite().all()
            # Report invalid predictions at the move boundary, but keep indices safe
            # until then: a NaN PUCT score could otherwise select an invalid edge.
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            legal = ~states[:, :, -1].bool().any(dim=1)
            # Terminal full boards must not produce an all-infinite softmax row.
            scores = logits.masked_fill(~legal, -torch.inf)
            scores = torch.where(legal.any(dim=1, keepdim=True), scores, torch.zeros_like(scores))
            priors = torch.softmax(scores, dim=1).masked_fill(~legal, -1)
            self.priors[self.ids, nodes] = torch.where(mask[:, None], priors, self.priors[self.ids, nodes])
            self.expanded[self.ids, nodes] |= mask
        return values

    def _select_cpu(self):
        nodes = self.roots.clone()
        self.paths[:, 0] = nodes
        self.lengths.fill_(1)
        self.created.zero_()
        self.parents.copy_(nodes)
        self.actions.zero_()
        for depth in range(self.cells):
            active = self.expanded[self.ids, nodes] & self.terminal[self.ids, nodes].isnan() & ~self.created
            if not active.any():
                break
            children = self.children[self.ids, nodes]
            safe = children.clamp_min(0)
            visits = self.visits[self.ids[:, None], safe] * (children >= 0)
            sums = self.sums[self.ids[:, None], safe] * (children >= 0)
            priors = self.priors[self.ids, nodes]
            scale = self.visits[self.ids, nodes].clamp_min(1).float().sqrt()
            scores = -sums / visits.clamp_min(1) + self.c_puct * priors * scale[:, None] / (1 + visits)
            action = scores.masked_fill(priors < 0, -torch.inf).argmax(dim=1)
            child = children[self.ids, action]
            new = active & (child < 0)
            child = torch.where(new, self.counts, child).clamp_min(0)
            self.children[self.ids, nodes, action] = torch.where(active, child, children[self.ids, action])
            self.counts += new
            self.parents = torch.where(active, nodes, self.parents)
            self.actions = torch.where(active, action, self.actions)
            nodes = torch.where(active, child, nodes)
            self.paths[:, depth + 1] = nodes
            self.lengths += active
            self.created |= new
        self.leaves.copy_(nodes)

    def _select(self):
        if self.kernels is None:
            self._select_cpu()
        else:
            self.kernels.select_leaves[(self.games,)](
                self.roots, self.children, self.priors, self.visits, self.sums, self.expanded,
                self.terminal, self.counts, self.paths, self.lengths, self.leaves, self.parents,
                self.actions, self.created, N=self.capacity, C=self.cols, D=self.cells + 1,
                BLOCK=2 ** math.ceil(math.log2(self.cols)), CPUCT=self.c_puct, num_warps=1)

    def _materialize(self):
        parent = self.boards[self.ids, self.parents]
        board = parent.flip(1).clone()
        heights = parent.sum(dim=(1, 2))
        row = heights[self.ids, self.actions].long().clamp_max(self.config.rows - 1)
        board[self.ids, 1, row, self.actions] = 1
        won = board[:, 1].flatten(1)[:, self.lines].bool().all(dim=2).any(dim=1)
        full = board.sum(dim=(1, 2, 3)) == self.cells
        terminal = torch.where(won, -torch.ones(self.games, device=self.device),
                               torch.where(full, torch.zeros(self.games, device=self.device), torch.nan))
        self.boards[self.ids, self.leaves] = torch.where(
            self.created[:, None, None, None], board, self.boards[self.ids, self.leaves])
        self.terminal[self.ids, self.leaves] = torch.where(
            self.created, terminal, self.terminal[self.ids, self.leaves])

    def _backup(self, values):
        if self.kernels is not None:
            self.kernels.backup_values[(self.games,)](self.paths, self.lengths, values,
                self.visits, self.sums, N=self.capacity, D=self.cells + 1, num_warps=1)
            return
        for depth in range(int(self.lengths.max())):
            valid = depth < self.lengths
            nodes = self.paths[:, depth]
            signed = values * torch.where((self.lengths - 1 - depth) % 2 == 0, 1, -1)
            self.visits[self.ids, nodes] += valid
            self.sums[self.ids, nodes] += signed * valid

    @torch.inference_mode()
    def search(self, add_noise=False):
        # Conservative host bound guarantees kernels never overrun their arrays.
        if 1 + (self.rounds + 1) * self.simulations > self.capacity:
            raise RuntimeError("Search node limit exhausted; increase --max-tree-nodes")
        self.rounds += 1
        mask = ~self.expanded[self.ids, self.roots] & self.terminal[self.ids, self.roots].isnan()
        self._evaluate(self.roots, mask)
        base = self.priors[self.ids, self.roots].clone()
        if add_noise:
            # One host boundary per move; legal moves are known from real games.
            noise = np.zeros((self.games, self.cols), dtype=np.float32)
            for game, position in enumerate(self.positions):
                legal = position.legal_moves()
                if legal:
                    noise[game, legal] = self.rng.dirichlet([self.alpha] * len(legal))
            mixed = (1 - self.noise_fraction) * base + self.noise_fraction * torch.as_tensor(noise, device=self.device)
            self.priors[self.ids, self.roots] = torch.where(base >= 0, mixed, base)
        for _ in range(self.simulations):
            with self.profile.phase("traversal"):
                self._select()
            with self.profile.phase("expansion"):
                self._materialize()
            mask = ~self.expanded[self.ids, self.leaves] & self.terminal[self.ids, self.leaves].isnan()
            predictions = self._evaluate(self.leaves, mask)
            terminal = self.terminal[self.ids, self.leaves]
            values = torch.where(terminal.isnan(), predictions, terminal)
            with self.profile.phase("backup"):
                self._backup(values)
        self.priors[self.ids, self.roots] = base
        with self.profile.phase("results_transfer"):
            children = self.children[self.ids, self.roots]
            visits = (self.visits[self.ids[:, None], children.clamp_min(0)] * (children >= 0)).cpu().numpy()
            values = (self.sums[self.ids, self.roots] / self.visits[self.ids, self.roots].clamp_min(1)).cpu().numpy()
            if not self.finite.item():
                raise RuntimeError("Nonfinite policy/value predictions")
        results = []
        for game, (counts, value) in enumerate(zip(visits, values)):
            terminal = self.positions[game].terminal_value()
            if terminal is not None:
                counts = np.zeros(self.cols, dtype=np.int32)
                value = terminal
            policy = counts / counts.sum() if counts.sum() else np.zeros(self.cols)
            results.append(SearchResult(policy.astype(np.float32), counts.astype(np.int64), float(value)))
        return results

    @torch.inference_mode()
    def advance(self, actions):
        """None keeps completed games in a masked fixed-size batch."""
        if len(actions) != self.games:
            raise ValueError("One action is required per game")
        positions = [position if action is None else position.play(action)
                     for position, action in zip(self.positions, actions)]
        active = torch.tensor([a is not None for a in actions], device=self.device)
        cols = torch.tensor([a or 0 for a in actions], device=self.device)
        roots = self.children[self.ids, self.roots, cols]
        if (active & (roots < 0)).any():
            raise ValueError("Advance requires a visited search action")
        self.roots = torch.where(active, roots, self.roots)
        self.positions = positions
