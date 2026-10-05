"""Run Triton's actual kernel bodies on CPU with TRITON_INTERPRET=1.

This checks pointer arithmetic and state transitions without a CUDA device;
it complements rather than replaces tests of compiled CUDA kernels.
"""
import importlib.util
import os
import re
import unittest

import numpy as np
import torch

from alphazero import Position
from array_mcts import ArrayMCTS
from test_array_mcts import FixedNet, position_after


@unittest.skipUnless(importlib.util.find_spec("triton") and os.environ.get("TRITON_INTERPRET") != "1",
                     "Triton compiler required (without interpreter mode)")
class KernelCompilationTests(unittest.TestCase):
    def test_t4_compilation_preserves_loop_carried_parent(self):
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from search_kernels import select_leaves
        signature = {name: "*i64" for name in
                     ("roots", "children", "counts", "paths", "lengths", "leaves", "parents", "actions")}
        signature.update({name: "*fp32" for name in ("priors", "sums", "terminal")})
        signature.update(visits="*i32", expanded="*i1", created="*i1")
        source = ASTSource(select_leaves, signature,
                           constexprs=dict(N=5377, C=7, D=43, BLOCK=8, CPUCT=1.5))
        compiled = triton.compile(source, target=GPUTarget("cuda", 75, 32), options=dict(num_warps=1))
        ir = compiled.asm["ttir"]
        function = next(line for line in ir.splitlines() if "tt.func public @select_leaves" in line)
        parent_argument = re.findall(r"(%[\w]+): !tt.ptr", function)[11]
        store = re.search(r"(%[\w]+) = tt.addptr " + re.escape(parent_argument)
                          + r",[^\n]+\n\s*tt.store \1, (%[\w]+)#(\d+)", ir)
        self.assertIsNotNone(store, "Parent must be stored from a traversal-loop result, not the initial root")
        self.assertRegex(ir, re.escape(store.group(2)) + r":\d+ = scf.while")


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1" and importlib.util.find_spec("triton"),
                     "Set TRITON_INTERPRET=1 with Triton installed")
class KernelInterpreterTests(unittest.TestCase):
    def test_kernel_search_matches_tensor_reference(self):
        import search_kernels
        model = FixedNet(value=0.37)
        positions = [Position.empty(), position_after([0, 6, 1, 6, 2, 5]),
                     position_after([0, 3, 0, 3, 1, 3]), position_after([0] * 6)]
        reference = ArrayMCTS(model, positions, simulations=128)
        interpreted = ArrayMCTS(model, positions, simulations=128)
        interpreted.kernels = search_kernels
        for step in range(2):
            left, right = reference.search(), interpreted.search()
            for game, (a, b) in enumerate(zip(left, right)):
                np.testing.assert_array_equal(a.visits, b.visits, err_msg=f"game={game}, move={step}")
                self.assertAlmostEqual(a.value, b.value, places=5)
            torch.testing.assert_close(reference.counts, interpreted.counts)
            torch.testing.assert_close(reference.children, interpreted.children)
            torch.testing.assert_close(reference.boards, interpreted.boards)
            if step == 0:
                actions = [result.action() for result in left]
                reference.advance(actions)
                interpreted.advance(actions)


if __name__ == "__main__":
    unittest.main()
