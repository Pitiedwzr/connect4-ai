# Connect 4 Agent

A Double DQN agent with a dueling convolutional network, minimax opponents,
and a Pygame interface. Python 3.12 or newer is required.

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
