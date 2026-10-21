# Playing-strength experiments

## Loss traces and checkpoint selection

The iteration-1900 `model/experiments/proven_win/inference.eqx` is the current
working reference. Its selected `latest_best.eqx` is an earlier iteration;
compare snapshots using several opponents and validation seeds before promoting
a replacement.

Save replayable losses, concentrating first on second-seat minimax games:

```powershell
uv run benchmark.py --agent equinox --model model/experiments/proven_win/inference.eqx --simulations 128 --proven-win-priority --games 50 --seed 20261023 --depths 2 4 5 --trace-output logs/proven_win_losses_seed20261023.jsonl --trace-opponents minimax_2 minimax_4 minimax_5 --trace-seat second --output benchmarks/proven_win_losses_seed20261023.json
uv run analyze_traces.py logs/proven_win_losses_seed20261023.jsonl --output logs/proven_win_loss_summary.json
```

Tracing defaults to losses. `--trace-games all` also saves wins and draws;
`--trace-seat both` and omission of `--trace-opponents` capture the full suite.
The JSONL starts with a run record containing settings, filters and Equinox
checkpoint hashes. Each subsequent game includes stable game/opening identifiers,
the exact opening prefix, candidate seat, every move and pre-move board, winner,
and final board. Row zero is the bottom row; actions are zero-based columns.
The trace file must be new, so an existing experiment is never overwritten.
Results and seeded openings remain identical with tracing enabled.

Candidate decisions include exact immediate winning/safe actions and flags for
missed immediate wins or avoidable immediate opponent wins. Equinox search
decisions additionally record root priors, visits, Q estimates, network/root
values, selected policy/action, rewards, discounts, and visited terminal-win
proofs. Unvisited Q entries are not evidence; a Q estimate of +1 is not a proof.
Raw mode records network logits and value. Other agent types retain moves and
exact tactical annotations without Equinox statistics. Diagnostics add export
and annotation overhead: use runs without tracing for latency comparisons.
The summary locates one-ply mistakes by game ID and ply; losses without those
flags still require inspection for deeper threats or value/search disagreement.

Existing checkpoints can be selected offline on CPU, without further training:

```powershell
uv run select_jax_checkpoint.py --models model/experiments/proven_win/latest_best.eqx model/experiments/proven_win/inference.eqx model/experiments/no_proven_win/latest_best.eqx model/experiments/no_proven_win/inference.eqx --seeds 12345 12346 12347 --depths 4 5 --games 3 --opponent-model model/experiments/larger_refined/latest_best.eqx --report logs/checkpoint_selection.json --best-output model/experiments/selected/inference.eqx
```

This evaluates raw and searched play on the same seeds and openings. Selection
uses the equal-weight mean search score against configured minimax depths and
the optional frozen model. Random/tactical-random results are reported but are
excluded when minimax opponents exist. Without minimax depths, tactical random
is the fallback. Ties use the weakest opponent's pooled score, then the raw-policy
mean against the same opponents; exact ties preserve the earlier candidate.
Per-seed W/D/L and checkpoint hashes are recorded. Report/export paths must be
new and separate from source checkpoints. The exported winner records the
effective evaluation search settings; source files are preserved.

New training runs use `--eval-selection suite`: by default three validation
seeds (`--eval-seed` and the next two) and depths 2/4/5. Each seed runs
`--eval-games` games per seat, opponent and raw/search mode. This costs more CPU
time than the old single-seed evaluation; `--eval-games 3` is a reasonable cheap
shortlisting setting, followed by a larger independent promotion benchmark.
The optional `--eval-opponent-model` adds a frozen Equinox reference with the
same search settings. Its content hash is part of the evaluation signature;
changing the file during training raises an error. Selection history resets
when seeds, opponents, search controls, or the selection rule change.

To continue on the original T4 x 2 host with broader validation:

```powershell
uv run train_jax.py --platform gpu --devices 2 --resume model/experiments/proven_win/latest.eqx --iterations 500 --eval-selection suite --eval-seeds 12345 12346 12347 --eval-depths 4 5 --eval-games 3 --eval-opponent-model model/experiments/proven_win/inference.eqx --output model/experiments/validated/latest.eqx --export model/experiments/validated/inference.eqx --log logs/jax_validated.jsonl
```

Older checkpoints retain their saved single-seed/deepest-opponent selection on
resume unless explicitly upgraded. `--eval-selection deepest` retains that rule
for new runs too. Use fresh held-out seeds for final promotion, separate from
the validation seeds used to select checkpoints. The commands above prepare
experiments; no new training or full opponent benchmarks were run to implement
this tooling.

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

`latest_best.eqx` contains the model selected by the configured validation rule.
New runs use the multi-seed suite mean, weakest-opponent score, and raw-policy
mean described above. Legacy `--eval-selection deepest` uses search-assisted
score against the deepest configured minimax opponent, with raw-policy score
breaking ties. Score is `(wins + 0.5*draws)/games`, combined across both seats.
Without minimax opponents, tactical random is the fallback. Exact ties preserve
the earlier checkpoint. The selection history
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
