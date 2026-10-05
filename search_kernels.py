"""Optional Triton kernels. Imported only by the CUDA search backend."""
import triton
import triton.language as tl


@triton.jit
def select_leaves(roots, children, priors, visits, sums, expanded, terminal,
                  counts, paths, lengths, leaves, parents, actions, created,
                  N: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
                  BLOCK: tl.constexpr, CPUCT: tl.constexpr):
    game = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    node = tl.load(roots + game)
    depth = 0
    # Keep this as a distinct SSA value. Triton can lose a loop-carried variable
    # initialized as a direct alias of another one: the compiled kernel then
    # returns the initial root as parent even after traversing deeper nodes.
    parent = node + 0
    action = 0
    new = False
    tl.store(paths + game * D, node)
    outcome = tl.load(terminal + game * N + node)
    running = tl.load(expanded + game * N + node) & (outcome != outcome)
    while running & (depth < D - 1):
        edge = (game * N + node) * C + cols
        child = tl.load(children + edge, cols < C, other=-1)
        prior = tl.load(priors + edge, cols < C, other=0.0)
        child_visits = tl.load(visits + game * N + tl.maximum(child, 0),
                               (cols < C) & (child >= 0), other=0)
        child_sum = tl.load(sums + game * N + tl.maximum(child, 0),
                            (cols < C) & (child >= 0), other=0.0)
        scale = tl.sqrt(tl.maximum(tl.load(visits + game * N + node), 1).to(tl.float32))
        score = -child_sum / tl.maximum(child_visits, 1) + CPUCT * prior * scale / (1 + child_visits)
        score = tl.where((cols < C) & (prior >= 0), score, -float('inf'))
        best = tl.max(score, 0)
        action = tl.min(tl.where(score == best, cols, BLOCK), 0)
        parent = node + 0
        node = tl.load(children + (game * N + parent) * C + action)
        new = node < 0
        if new:
            node = tl.load(counts + game)
            tl.store(counts + game, node + 1)
            tl.store(children + (game * N + parent) * C + action, node)
        depth += 1
        tl.store(paths + game * D + depth, node)
        outcome = tl.load(terminal + game * N + node)
        running = (~new) & tl.load(expanded + game * N + node) & (outcome != outcome)
    tl.store(lengths + game, depth + 1)
    tl.store(leaves + game, node)
    tl.store(parents + game, parent)
    tl.store(actions + game, action)
    tl.store(created + game, new)


@triton.jit
def backup_values(paths, lengths, values, visits, sums,
                  N: tl.constexpr, D: tl.constexpr):
    game = tl.program_id(0)
    depth = tl.load(lengths + game) - 1
    value = tl.load(values + game)
    while depth >= 0:
        node = tl.load(paths + game * D + depth)
        offset = game * N + node
        tl.store(visits + offset, tl.load(visits + offset) + 1)
        tl.store(sums + offset, tl.load(sums + offset) + value)
        value = -value
        depth -= 1
