# Playing-strength experiments

Start with the existing 32-channel/two-block checkpoint. Change search, training
coverage, and learning rate before attributing the remaining weaknesses to model
capacity. These settings are experiments; their effect needs opponent benchmarks.

## Proved-win and search-scoring experiments

Start by benchmarking the frozen refined model with `--proven-win-priority`.
This opt-in setting prefers root moves already explored by MCTS whose exact
game transition has reward +1 and zero discount. A neural value of +1 does not
establish a proof. Other positions retain their usual search action/targets.
Multiple proved wins receive a normalized target distribution from their root
priors, and the largest prior chooses the recommended move. Target calculation
uses logits so even extremely unlikely winning moves receive finite labels.
The rule applies to both Gumbel and PUCT, but PUCT can miss a low-prior move
without visiting it. Terminal batch slots and illegal/unvisited edges cannot
establish proofs. This implements immediate terminal-win priority, not general
propagation of deeper win/loss proofs.

For a controlled comparison, use the same checkpoint for both players:

```bash
uv run benchmark.py --agent equinox \
  --model model/experiments/larger_refined/latest_best.eqx \
  --proven-win-priority --simulations 128 \
  --games 50 --seed 20261008 --depths 2 4 5 \
  --opponent-model model/experiments/larger_refined/latest_best.eqx \
  --opponent-simulations 128 --no-opponent-proven-win-priority \
  --opponent-prior-temperature 1 --opponent-gumbel-q-scale 0.1 \
  --output benchmarks/proven_win_seed20261008.json
```

Repeat over several seeds, including existing benchmark seeds for paired opening
comparisons. Candidate and opponent search controls are independent; a candidate
override does not enable proved-win priority or change scoring for the opponent.
JSON reports record both effective configurations. Opponent policy can be set
with `--opponent-search-policy`; absent that flag, an explicit `--search-policy`
continues to apply to both players, otherwise each checkpoint supplies its policy.

The default controls preserve existing checkpoints:

| Flag | Default for old checkpoints | Effect |
|---|---:|---|
| `--proven-win-priority` | Off | Prefer proved immediate wins and correct self-play targets |
| `--prior-temperature` | 1.0 | Divide root and leaf policy logits; greater values flatten priors |
| `--gumbel-q-scale` | 0.1 | Scale Gumbel's completed Q values; PUCT ignores this setting |

Temperature and Q scale must be finite and positive. These are search parameters;
raw prediction remains unchanged. `--c-puct` only tunes PUCT. Proved-win priority
and scoring controls are supported by `train_jax.py`, `play_jax.py`,
`benchmark.py`, and `benchmark_jax.py`. Inference commands inherit checkpoint
settings unless explicitly overridden; use `--no-proven-win-priority` to disable
a saved setting. New checkpoints and inference exports record all controls.
The GUI also inherits them automatically when loading such a checkpoint.

The frozen-model diagnostic found that prior temperature 2 and Q scale 0.5
improved immediate-win detection but reduced safe-reply accuracy. Keep 1/0.1
as the reference while benchmarking scoring ablations; no stronger defaults
have been established by opponent matches. Proved-win priority is opt-in even
with the existing `improve`/`larger` presets.

After validating the frozen-model variant, continue on the same T4 x 2 host:

```bash
uv run train_jax.py --platform gpu --devices 2 \
  --resume model/experiments/larger_refined/latest.eqx \
  --preset improve --proven-win-priority --iterations 500 \
  --prior-temperature 1 --gumbel-q-scale 0.1 \
  --output model/experiments/proven_win/latest.eqx \
  --export model/experiments/proven_win/inference.eqx \
  --log logs/jax_proven_win.jsonl
```

This keeps the loaded 64-channel/two-block architecture and uses learning rate
0.0003. Resume restores optimizer/replay/RNG state and the saved controls unless
explicitly overridden. Existing replay labels turn over gradually; `--init-from`
starts with weights and a fresh optimizer/replay. The new option corrects both
the self-play action and its policy target whenever a proved win is present.
PUCT's temperature sampling also stays within proved winning actions.

Periodic evaluation uses the same effective controls as self-play. Changing
controls starts a new best-model comparison; retain the previous reference for
held-out evaluation. The original refined checkpoint and benchmarks remain
preserved by the separate output paths above.

```bash
uv run benchmark_jax.py --model model/experiments/larger_refined/latest_best.eqx \
  --proven-win-priority --simulations 64 128 256 --repeats 30
uv run play_jax.py --model model/experiments/larger_refined/latest_best.eqx \
  --proven-win-priority --simulations 128
```

## Continue the trained model on T4 x 2

On the Linux training host, use the existing CUDA installation profile and run:

```bash
uv run train_jax.py --platform gpu --devices 2 \
  --resume model/connect4_alphazero.eqx --preset improve --iterations 1000 \
  --output model/experiments/improved/latest.eqx \
  --export model/experiments/improved/inference.eqx \
  --checkpoint-dir model/experiments/improved/checkpoints --keep-checkpoints 0 \
  --log logs/jax_improved.jsonl
```

`--preset improve` selects:

| Setting | Value |
|---|---|
| Search policy | Gumbel |
| Self-play simulations | 128 |
| Learning rate | 0.0003 |
| Games with random opening prefixes | 20% |
| Prefix length | Uniformly sampled from 1–8 plies |
| Checkpoint interval | 100 iterations |
| Evaluation interval | 100 iterations, plus final iteration |
| Evaluation | Raw policy and CPU search, 128 simulations |
| Opponents | Random, tactical random, minimax depths 2 and 4 |
| Evaluation games | 10 per seat, per opponent, per mode |

Explicit flags override the preset. The saved batch size, games per device,
optimizer moments, replay, and RNG restore on resume. Gumbel targets replace
the older PUCT targets as replay turns over. Use `--init-from` instead if you want
fresh optimizer/replay/RNG state; it preserves the checkpoint's architecture.

The original checkpoint and log are preserved by the separate output paths above.
Resume requires the original device count/platform. For a move to TPU, use
`--init-from` with `--platform tpu --devices 8 --precision bf16` instead of resume.

## Search and opening semantics

`--search-policy puct` retains the original visit-target search. In PUCT self-play,
Dirichlet noise and `--temperature-moves` retain their original behavior.

`--search-policy gumbel` considers all legal root actions using sequential halving.
The simulation budget must be at least the number of columns. Self-play uses
mctx's recommended Gumbel action and its improved policy distribution as the
training target. It does not replace that action with the most visited action or
the largest target probability. Gumbel noise supplies self-play exploration;
Dirichlet and temperature settings apply only to PUCT. Evaluation sets Gumbel
noise to zero and uses its recommended action deterministically.

Root exploration improves coverage; extremely small priors or inaccurate values
can still cause tactical mistakes, even after a winning move is explored.

`--opening-fraction 0.2 --opening-plies 8` starts some games from random legal
prefixes. Any prefix reaching a win or draw is discarded and replaced by the
empty board. Logs report the actual accepted prefix count and mean prefix length.
Prefix moves receive no training labels. Subsequent positions use their actual
side to move for both encoding and final-outcome targets. No heuristic labels,
forced opening, or tactical filter is introduced.

CPU players automatically use the policy recorded in the checkpoint. Old weights
default to PUCT. Override the policy explicitly for a controlled comparison:

```bash
uv run benchmark.py --agent equinox --model model/connect4_alphazero.eqx --search-policy puct --simulations 128 --games 50
uv run benchmark.py --agent equinox --model model/connect4_alphazero.eqx --search-policy gumbel --simulations 128 --games 50
```

## Snapshots, evaluation, and best-model selection

`--checkpoint-dir` retains full resumable snapshots named
`latest.iter-0001100.eqx`, etc. The default retention is the latest ten snapshots;
`--keep-checkpoints 0` keeps all. Retention only removes snapshots matching the
current output stem, preserving unrelated files. Presets create a snapshot
directory beside the output if no directory is supplied.

The latest output remains a resumable training checkpoint. Every evaluation also
writes an iteration report in `latest_evaluations/` beside it, and includes the
report and evaluation time in JSONL. CPU evaluation may add substantial wall time;
its move latency excludes compilation. Disable it with `--eval-every 0`, or use
`--eval-games 5` for a cheaper initial run.

`latest_best.eqx` contains the strongest evaluated inference model. Selection uses
search-assisted score against the deepest configured minimax opponent, with raw
policy score against that opponent breaking ties. Score is `(wins + 0.5*draws)/games`,
combined across both seats. Without minimax opponents, selection uses tactical
random. Equal scores preserve the earlier checkpoint. The selection history
restores on resume when the evaluation settings and best-output path match.
Changing evaluation settings starts a new comparison. `--best-output` overrides
the default path.

The best checkpoint is inference-only. To continue its training, use `--init-from`,
or resume its corresponding full iteration snapshot with a new `--output` path.
Use a different evaluation seed and more games for the final comparison, since
selecting many checkpoints on one small suite can overfit that suite:

```bash
uv run benchmark.py --agent equinox --model model/experiments/improved/latest_best.eqx --simulations 128 --games 50 --seed 20261006
uv run benchmark.py --agent equinox --model model/experiments/improved/latest_best.eqx --raw --games 50 --seed 20261006
uv run benchmark_jax.py --model model/experiments/improved/latest_best.eqx --simulations 64 128 256 --repeats 30
```

## Separate capacity experiment

The 64-channel/two-block model has 152,855 parameters. It requires a fresh training
run; the 32-channel checkpoint cannot be widened by resume or init-from.

```bash
uv run train_jax.py --platform gpu --devices 2 --preset larger --iterations 1000 \
  --games-per-device 32 --batch-size 512 \
  --output model/experiments/larger/latest.eqx \
  --export model/experiments/larger/inference.eqx \
  --checkpoint-dir model/experiments/larger/checkpoints --keep-checkpoints 0 \
  --log logs/jax_larger.jsonl
```

The larger preset uses the same search/diversity/evaluation settings, with 64
channels, two blocks, and initial learning rate 0.001. For a capacity comparison,
run a fresh 32-channel control with the same settings by overriding `--channels 32`
and using separate output/log directories. Compare playing strength, games and
positions collected, elapsed training time, and CPU move latency.

To isolate the other improvements, run separate continuation experiments:

1. PUCT, 128 simulations, learning rate 0.0003, no prefixes.
2. Gumbel with the same settings, no prefixes.
3. Gumbel with 20% random prefixes.

Use explicit `--search-policy` and `--opening-fraction` overrides with the improve
preset, and distinct directories for each run. Do not infer improvement from
self-play draw counts or training losses alone.
