# Connect 4 Agent

Connect 4 agents with a Pygame interface, minimax opponents, and an Equinox/JAX
AlphaZero training and CPU inference path using mctx. The earlier Double DQN
and PyTorch AlphaZero implementations remain available for comparison.
Python 3.12 or newer is required.

## Equinox + mctx training and CPU inference

For the next strength experiments, see [EXPERIMENTS.md](EXPERIMENTS.md).
It includes Gumbel search, random legal opening prefixes, reduced-learning-rate
continuation, retained snapshots, periodic raw/search evaluation, and a separate
64-channel capacity experiment. Checkpoints retain their saved search policy;
older checkpoints without policy metadata default to PUCT for CPU inference.
The guide also covers opt-in `--proven-win-priority`, policy prior temperature,
and Gumbel Q-value scaling, with independent controls for benchmark opponents.

`train_jax.py` uses a residual Equinox policy/value network, Optax AdamW, and
exact game transitions inside mctx PUCT or Gumbel search. Board updates, leaf inference,
tree search, and action selection stay inside compiled JAX self-play. Each
device collects its own games; learner gradients are averaged across devices.
One host is supported, including eight local v5e devices or two local T4 GPUs.

The default network is 32 channels with two residual blocks and GroupNorm.
`--channels 64 --blocks 4` reproduces the legacy AlphaZero architecture.
Both seats produce examples in their own perspective; final win/loss/draw
targets are +1/-1/0. Nonterminal search edges use discount -1 to reverse player
perspective; winning edges reward the acting player +1 and terminate backup.
Only exact rules and legal actions guide search. No minimax teacher, forced
opening, or tactical filter is used.

The dependency versions tested locally are locked in `uv.lock`. CPU installation
works on Windows and Linux:

```powershell
uv sync --locked
uv run train_jax.py --cpu --iterations 10 --games-per-device 2 --simulations 16 --batch-size 32 --warmup-positions 128 --updates-per-iteration 4 --output model/connect4_alphazero.eqx --export model/connect4_alphazero_inference.eqx --log logs/jax_cpu.jsonl
```

These are experiment settings, not a promise of trained playing strength.
The first collection and first learner update include compilation; the JSONL
marks those phases separately. `--batch-size` is global, while
`--games-per-device` is local. Batch size must divide evenly across devices.
Self-play outcome counts are not evidence of improvement against opponents.
On CPU hosts with limited memory, setting `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1` before launching Python bounds auxiliary numerical thread
pools, especially when running the test suite's subprocess checks.

### v5e-8 and T4 x 2

Accelerator extras require Linux. Select exactly one extra. JAX CUDA 12 requires
a Linux NVIDIA driver >=525; CUDA 13 requires >=580. The T4 supports either
family when its driver is compatible. Check the installed runtime before training:

```bash
# TPU runtime with eight local devices
uv sync --locked --extra tpu
uv run python -c "import jax; print(jax.local_devices())"
uv run train_jax.py --platform tpu --devices 8 --games-per-device 64 --simulations 64 --batch-size 1024 --precision bf16 --iterations 1000 --output model/connect4_alphazero.eqx --export model/connect4_alphazero_inference.eqx --log logs/jax_tpu.jsonl

# Two T4 GPUs; use cuda13 instead if the host driver requires that profile
uv sync --locked --extra cuda12
uv run python -c "import jax; print(jax.local_devices())"
uv run train_jax.py --platform gpu --devices 2 --games-per-device 32 --simulations 64 --batch-size 512 --precision fp32 --iterations 1000 --output model/connect4_alphazero.eqx --export model/connect4_alphazero_inference.eqx --log logs/jax_t4.jsonl
```

Before a long accelerator run, execute a two-iteration smoke run with the same
platform/device options, `--games-per-device 2 --simulations 4 --batch-size 32
--warmup-positions 32 --updates-per-iteration 1`, and a separate output path.
Verify that its exported checkpoint loads with CPU inference. Local CPU checks
validate sharding semantics but do not establish accelerator compatibility or speed.

FP32 is the T4 baseline. BF16 convolution compute is available on TPU; master
parameters, GroupNorm, dense heads, losses, optimizer state, and search statistics
remain FP32. FP16 training is deferred until loss scaling is implemented and
validated. Compare FP32 and BF16 training on the TPU before choosing precision.

`--search-memory-mb` defaults to 512 MiB per device and checks persistent mctx
tree arrays before collection. Neural activations and compiler temporaries are
additional. Trees are rebuilt each move, so their capacity is simulations + 1;
there is currently no subtree reuse or transposition cache. The earlier cached
Python implementation may therefore remain competitive for small CPU searches.

Replay uses a bounded host ring buffer. Horizontal reflection is applied when
sampling, including reflection of policy columns. Weights remain fixed during
each collection phase, and gradients do not pass through MCTS.

### Checkpoints, resume, and migration

The `.eqx` files are versioned ZIP bundles containing JSON architecture metadata
and Equinox model leaves. Training bundles also contain optimizer leaves, replay,
RNG streams, and counters. Writes atomically replace complete checkpoints;
inference loading reads only metadata and model weights and needs no PyTorch.

```powershell
uv run train_jax.py --cpu --resume model/connect4_alphazero.eqx --iterations 10
uv run train_jax.py --cpu --init-from model/connect4_alphazero_inference.eqx --iterations 10
uv run convert_checkpoint_jax.py model/legacy_alphazero.pth model/imported_alphazero.eqx
```

Resume restores saved training options unless explicitly overridden, and
`--iterations` means additional iterations. Preserve device count, platform,
replay capacity, and all training settings for exact continuation. Changing
settings intentionally changes the experiment. For a new hardware configuration,
use `--init-from` to retain weights and start fresh optimizer/replay/RNG streams.
Loading existing weights fixes board and network dimensions. Legacy conversion
checks forward-output agreement before saving; legacy optimizer and replay are
not converted. DQN checkpoints cannot be converted to policy/value models.

### CPU play and evaluation

```powershell
uv run play_jax.py --model model/connect4_alphazero_inference.eqx --simulations 128
uv run play_jax.py --model model/connect4_alphazero_inference.eqx --raw
uv run benchmark.py --agent equinox --model model/connect4_alphazero_inference.eqx --games 50 --simulations 128
uv run benchmark.py --agent equinox --model model/connect4_alphazero_inference.eqx --games 50 --raw
uv run benchmark_jax.py --model model/connect4_alphazero_inference.eqx --simulations 32 64 128 256 --repeats 30
```

The Equinox agent always places its model and search on CPU. Evaluation disables
root noise and chooses the most visited legal move. Raw mode chooses legal policy
logits directly. `benchmark_jax.py` reports warmup separately from p50/p95 move
latency on seeded legal opening prefixes. Compare playing strength at equal
thinking time, since simulation counts are not equivalent across implementations.
The baseline opponent benchmark still uses standard 6x7 Connect 4; training and
terminal play support larger boards.

The GUI's existing **AlphaZero Policy + Search** choice prefers
`model/connect4_alphazero.eqx` when present, and otherwise uses the legacy `.pth`
checkpoint. Both the player and advantage bar support Equinox weights. The first
search compiles in the existing background AI worker. The bar remains a position
estimate rather than calibrated win odds. Restart after replacing weights.

To compare accelerator collection throughput without training:

```bash
uv run benchmark_jax.py --collect --platform tpu --devices 8 --games-per-device 64 --simulations 64 128 --repeats 3
uv run benchmark_jax.py --collect --platform gpu --devices 2 --games-per-device 32 --simulations 64 128 --repeats 3
```

Without `--model`, these benchmarks use randomly initialized 32-channel/two-block
weights. Throughput excludes one warmup collection, but host result transfer is
included. Hardware selection should follow measured throughput and playing
strength gained per hour, rather than peak accelerator compute.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
uv run python -m unittest discover -s tests -v
```

The new tests cover reference rule parity (including boards exceeding 64 cells),
terminal rewards and value signs, mctx wins/blocks and descendant legality,
tree-memory accounting, reflection, training, checkpoint portability, exact CPU
resume, legacy weight conversion, GUI/benchmark integration, and gradient averaging
on two simulated CPU devices. Existing implementation tests remain in the suite.

## Setup and play

```powershell
uv sync
uv run gui.py
```

The GUI and CLI (`uv run game.py`) use `model/connect4_model_selfplay.pth`.
Checkpoints are generated locally and ignored by Git. The architecture remains
compatible with the existing self-play checkpoint. Missing or incompatible DQN
weights cause the GUI to select its displayed minimax opponent; direct DQN
loading raises an error rather than using random weights. GUI players are cached
so weights are loaded once per seat/configuration. Restart the GUI after replacing
a checkpoint.

## Tactical policy and learning targets

Training, GUI play, CLI play, and Double DQN bootstrap selection use one policy:

1. Open in the center on an empty board.
2. Take an immediate win.
3. Exclude moves allowing any immediate winning opponent reply when alternatives exist.
4. Exclude moves allowing an opponent reply that creates two winning columns,
   unless the agent can win immediately after that reply.
5. If every move loses, retain the best available set of legal moves.

The fork check is bounded tactical search, not a guarantee against deeper traps.
The center opening is an explicit rule, not evidence that the network learned it.
Replay stores the allowed next-action mask once when collecting each transition
and mirrors the mask with its board. Terminal transitions never bootstrap.
Transitions span the agent's move and the opponent's reply, so the next state is
again from the agent's turn; the ordinary positive discounted bootstrap is correct.

Soft target updates interpolate parameters and copy BatchNorm running statistics
and counters. Bootstrap estimates are bounded to `[-1, 1]`, the feasible return
range with terminal win/loss rewards. This limits numerical drift; it does not
make Q-values calibrated win probabilities.

## Train

[Accelerate supports single-CPU execution](https://github.com/huggingface/accelerate/blob/main/examples/README.md)
as well as GPU training. A direct invocation uses one process. CPU training needs
no CUDA and can be forced with `--cpu`:

```powershell
uv run train.py --cpu --pretrain-positions 2000 --output model/connect4_model_candidate.pth
```

Training starts with new weights. Search-labelled pretraining is optional
(`--pretrain-positions` defaults to zero); it uses legal positions from both seats
and mirrored examples to teach action preferences with cross entropy before RL.
Its labels come from bounded minimax, not a perfect solver, and its loss does not
define win probabilities. `--pretrain-depth` defaults to 3.

Self-play samples from a bounded pool of frozen historical opponents. Outside
forced moves, the opponent mixture is 60% historical DQN, 30% minimax, and 10%
tactical random. Minimax depth increases from 2 to 4 over training. Search also
guides some of the agent's training actions, with probability decaying from 0.25
to zero. Configure these with `--opponent-pool-size`, `--snapshot-every`,
`--minimax-max-depth`, and `--teacher-probability`.

For multiple GPUs:

```powershell
uv run accelerate launch --num_processes 2 train.py --pretrain-positions 2000 --output model/connect4_model_candidate.pth
```

`--episodes` is a global game budget (default 50,000). Distributed execution rounds
it up to a multiple of process count so every rank performs equal synchronized
steps. Epsilon decays by global games, not games on each rank. Search pretraining
positions are per process.

Frozen evaluation runs every 1,000 global games against random, tactical random,
and depth-2/depth-4 minimax, separately for both seats. Use `--eval-games` to set
games per seat/opponent or `--eval-every 0` to disable evaluation. Training logs
label their exploratory win rate separately from these frozen results.

## Evaluate a checkpoint

```powershell
uv run benchmark.py --cpu --model model/connect4_model_candidate.pth --games 50
uv run benchmark.py --cpu --model model/connect4_model_candidate.pth --games 50 --raw
```

The JSON report includes opening Q-values and W/D/L counts for each seat and fixed
opponent. Columns in diagnostics are zero-indexed: center is 3. `--raw` measures
the network without the tactical policy, so improvements in learned play can be
distinguished from improvements in the rules. Use the same seed and game count
when comparing checkpoints. The first game starts empty; subsequent games use
seeded legal opening prefixes of up to four plies, paired between red/yellow seats,
to exercise different positions against deterministic minimax. Configure this
with `--opening-plies`; zero evaluates only the empty-board matchup. Counts remain
a benchmark against these particular opponents and positions, not proof of perfect play.

After validating a candidate, copy it to `model/connect4_model_selfplay.pth` to
use it in the GUI. Training code improvements require retraining; existing weights
are not changed by updating the source.

The GUI's 0–100 advantage bar is a relative position estimate, **not win odds**.
It evaluates only the actual side to move. If a checkpoint predicts outside the
feasible Q range, the display falls back to a heuristic.

## Verify

```powershell
uv run python -m unittest discover -s tests -v
```

Tests cover wins, blocks, support traps, forks, mirrored replay masks, legal
fallbacks in lost positions, target-network normalization, constrained bootstraps,
terminal targets, reproducible evaluation, and a short Accelerate CPU training run
that saves and reloads a temporary checkpoint.

## AlphaZero experiments

`alphazero.py` implements a separate learning agent inspired by the
[AlphaZero paper](https://arxiv.org/pdf/1712.01815). Its network predicts a policy
over columns and a value for the side to move. PUCT search uses both predictions
to explore possible continuations. Self-play records the resulting visit
distribution and final outcome, and trains both heads on those targets.

This path uses **game rules only**: legal columns and exact wins/draws. It does not
use DQN's forced center opening, tactical filters, minimax labels, or handcrafted
position scores. An untrained model has no learned strategy; the purpose is to
observe strategy emerge through search-guided self-play.

The default network has four residual blocks with 64 channels, a column policy
head, and a `tanh` value head. GroupNorm keeps leaf evaluation independent of
the inference batch size. It is a compact experimental implementation, not a
reproduction of DeepMind's model scale or results.

### Start training

For a smaller CPU experiment:

```powershell
uv run train_alphazero.py --cpu --iterations 200 --games-per-iteration 8 --simulations 32 --channels 32 --blocks 2 --log model/alpha_training.jsonl
```

With a GPU, omit `--cpu`. The standard defaults use eight concurrent self-play
games, 64 new simulations per move, and 20 optimizer steps per iteration:

```powershell
uv run train_alphazero.py --iterations 1000 --log model/alpha_training.jsonl
```

The recommended starting budgets are experiments, not a guarantee of strong play
after a particular number of iterations. More simulations can improve search but
also slow data collection. Compare elapsed time, games, raw policy performance,
and search-assisted performance rather than loss alone.

One leaf from each active game is evaluated in a batch at each search simulation.
Each game reuses its tree between moves and caches network evaluations while
weights remain frozen. Root Dirichlet noise and sampling from visit counts provide
self-play exploration; after the first ten plies, action selection becomes greedy.
Training targets retain normalized visit distributions. Evaluation adds no noise
and chooses the most visited move.

Both players contribute training positions in their own perspective. Value targets
are `+1` for an eventual win, `-1` for a loss, and `0` for a draw, without discounting.
Each edge reverses the value sign during search backup. Horizontal reflection
augments boards and policy distributions. The loss combines masked policy cross
entropy and value MSE, with AdamW weight decay.

Self-play's red/yellow win counts describe its own games; they are **not** a measure
of improvement against an external opponent. Training emits one JSON record per
iteration with games, replay positions, policy/value loss, and elapsed time. Use
the benchmark to measure learned behavior and playing strength.

### Save and resume

The default checkpoint is `model/connect4_alphazero.pth`, separate from DQN.
It includes architecture/game configuration, weights, optimizer state, replay,
iteration/game counts, and RNG state. Checkpoints are written every ten iterations
and at normal completion; `--checkpoint-every` and `--output` are configurable.
An interrupted write preserves the last complete checkpoint.

```powershell
uv run train_alphazero.py --cpu --resume model/connect4_alphazero.pth --iterations 200 --log model/alpha_training.jsonl
```

`--iterations` means **additional** iterations on resume. The checkpoint's network
dimensions/channels/blocks are reused. Saved training settings are also reused
unless explicitly overridden on the command line. On one process, replay,
optimizer, and RNG restore support continuation of the same experiment.
Without an explicit `--output`, resume saves back to its input checkpoint.

Accelerate can distribute independent self-play groups and synchronize training:

```powershell
uv run accelerate launch --multi_gpu --num_processes 2 train_alphazero.py --iterations 1000
```

Games per iteration and replay capacity are per process; reported games are global.
Distributed resume restores weights and optimizer but starts fresh local replay
and RNG streams; it does not claim bit-for-bit continuation across ranks. CUDA and
multi-process execution require validation on the target hardware.

### Experimental accelerated search

`--search-backend python` remains the default and uses the original cached search.
`--search-backend cpu` is a tensor-array reference implementation, useful for
correctness comparisons; it is not a compiled CPU speed optimization.
`--search-backend cuda` keeps boards, tree statistics, leaf selection and backup
on the GPU. It requires CUDA PyTorch and Triton on Linux (including Kaggle).
Triton is loaded only when this backend is requested; CPU/Windows users do not
need an additional dependency.

The array backends allocate children lazily and retain each game's tree between
moves. GPU selection and backup use fused Triton kernels. Board transitions,
Connect-N detection and policy/value inference use batched device operations.
No prediction is returned to Python within a search simulation. Action selection,
exploration noise and replay generation still run on the CPU once per real move.
The network architecture and existing checkpoints are unchanged.

These backends use fixed inference batches, including masked completed games and
terminal leaves, and do not use the Python backend's transposition cache. This
avoids per-simulation host synchronization but can evaluate redundant positions.
Kernel launch overhead and the small network may still limit throughput. Compare
end-to-end speed before selecting CUDA search; a speedup is not guaranteed.

Run the optional CUDA correctness test on the GPU host before training:

```bash
uv run python -m unittest discover -s tests -p test_array_mcts.py -v
```

CUDA tests include per-simulation comparisons of allocation, parent/leaf indices,
board states, visits, and value backup. They are skipped when CUDA/Triton is
unavailable. An additional offline T4 compiler regression test checks that the
generated traversal loop preserves the immediate parent of each leaf:

```bash
uv run python -m unittest discover -s tests -p test_search_kernels.py -v
TRITON_INTERPRET=1 uv run python -m unittest discover -s tests -p test_search_kernels.py -v
```

The second command runs the actual kernel bodies in Triton's CPU interpreter.
Interpreter execution cannot catch every compiler issue, so it complements the
offline compilation and CUDA execution checks. Then benchmark with the
same frozen network, precision, concurrent games and search budget:

```bash
uv run benchmark_search.py --backends python cuda --games 32 --simulations 128 --mixed-precision fp16
uv run benchmark_search.py --backends python cuda --games 64 --simulations 128 --mixed-precision fp16
uv run benchmark_search.py --backends python cuda --games 128 --simulations 128 --mixed-precision fp16
```

Add `--model model/connect4_alphazero.pth` to compare your trained network.
The benchmark excludes warmup/JIT compilation and reports games per minute,
positions per second, durations and peak allocated CUDA memory. Different backends
can generate different games because of RNG consumption and floating-point ties;
position counts are reported to help interpret their throughput.

For training on one T4, if the CUDA tests pass and throughput improves:

```bash
uv run accelerate launch --num_processes 1 --mixed_precision fp16 train_alphazero.py --search-backend cuda --games-per-iteration 64 --simulations 128 --batch-size 256 --warmup-positions 4096 --replay-capacity 100000 --iterations 100 --log logs/alpha_cuda.jsonl
```

Resume an existing model with `--resume` and explicitly select the new backend.
Backend/memory options are stored in checkpoints and restored unless overridden.
`--max-tree-nodes` sets capacity per game; zero reserves enough nodes for a full
game at the requested simulation budget. An insufficient limit fails explicitly.
`--search-memory-mb` defaults to 512 MiB per process and bounds persistent tree
arrays; neural-network activations, temporary tensors and replay are additional.
The startup allocation fails with a memory estimate if that budget is exceeded.

Training JSONL now includes local collection/training durations and local games
per minute. Add `--profile-search` for detailed traversal, expansion, inference,
transfer and backup timings plus inference batch statistics. CUDA profiling
synchronizes at phase boundaries and slows collection; disable it for throughput
comparisons. Distributed phase timings describe rank zero, while game outcomes
and losses remain aggregated across ranks. CUDA kernels require validation on
your target GPU; local CPU tests do not establish their correctness or speed.

### Compare policy and search

```powershell
uv run benchmark.py --agent alphazero --cpu --games 50 --simulations 128
uv run benchmark.py --agent alphazero --cpu --games 50 --raw
uv run benchmark.py --agent alphazero --cpu --games 50 --simulations 128 --dqn-opponent model/connect4_model_selfplay.pth
```

AlphaZero uses the same paired opening suite and standard-board opponents as DQN.
Reports include raw opening policy/value predictions, search budget, mode, and
W/D/L by seat. `--raw` selects from legal policy logits without search. The optional
DQN opponent uses its existing tactical policy. For fair speed comparisons, measure
thinking time too: a simulation count is not equivalent to a minimax depth.

In the GUI, select **AlphaZero Policy + Search** and choose 64, 128, or 256
simulations. Its advantage bar uses the raw AlphaZero value head, not a calibrated
win probability or the DQN evaluator. Restart the GUI after updating a checkpoint.
If AlphaZero weights are unavailable, the GUI explicitly switches its displayed
opponent to minimax.

Terminal play is available without Pygame, including `--raw` policy play:

```powershell
uv run play_alphazero.py --cpu --simulations 128
```

### AlphaGo Analysis Studio (Web UI)

Launch the AlphaGo-style analysis studio modeled after DeepMind's evaluation interfaces (Lizzie / KaTrain / Chess.com Review):

```powershell
uv run python run_studio.py
```

Features included:
- **Real-Time Win Rate Advantage Gauge**: Dynamic Red vs. Yellow winning probabilities derived from MCTS root value heads.
- **Match Win Rate Timeline**: Full-game evaluation curve plotted along plies. Clicking any point scrubs the board to that move.
- **Candidate Move Overlays (HUD)**: Expected value (Q-value), MCTS visit count ($N$), visit share ($\pi$), and policy prior ($P$) displayed directly above each column.
- **Principal Variation (PV) & Ghost Piece Forecast**: Hovering over any candidate move previews the anticipated future continuation on the board with ghost discs and step numbers.
- **Game Review & Blunder Detection**: Automatic move quality classification (*Best Move*, *Excellent*, *Good*, *Inaccuracy*, *Mistake*, *Blunder*) with delta win rate drop.
- **What-If Sandbox**: Fork positions at any historical ply and test alternative moves without altering the main game record.
- **Deep Analyze**: On-demand deep search budget (up to 1,024+ simulations).


### Larger boards

The new environment and search use Python-integer bitboards and dimensions from
the checkpoint, including boards exceeding 64 bits. The network heads are sized
from configuration. For example:

```powershell
uv run train_alphazero.py --cpu --rows 8 --cols 9 --connect 4 --channels 32 --blocks 2 --simulations 32 --output model/alpha_8x9.pth
uv run play_alphazero.py --cpu --model model/alpha_8x9.pth
```

Changing board dimensions creates a new architecture/checkpoint; weights are not
automatically transferable between sizes. The existing GUI, DQN, and baseline
benchmark remain standard 6×7 Connect 4. Larger boards are supported by AlphaZero
self-play and terminal play. Learning plus bounded search can approximate good
decisions without solving the entire game, but does not guarantee superior play
or perfect solutions on larger boards.

AlphaZero tests additionally cover generic game rules, boards beyond 64 bits,
value-sign handling, wins/blocks found through search, batched leaf inference,
exploration noise, legal masks, both-player outcomes, mirrored targets, checkpoint
reload/resume, and raw/search benchmark execution.
