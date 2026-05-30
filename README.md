# Entity-Tracking Externalization Probe

**Does making a model write its working out — and in the right format — let it
solve longer, harder state-tracking problems than it can natively, with no
fine-tuning?** For small local models the answer here is a clear yes, and *how*
you externalize matters as much as whether you do.

This is a small, self-contained, reproducible test harness for that question. It
generates synthetic entity-tracking tasks (objects moving in and out of
containers), runs a model under different prompting strategies, grades the
answers deterministically, and measures the **competence frontier** — the
hardest problem each strategy can still solve — so you can see exactly how far a
given prompt format extends a given model's reach.

## Why you'd use it

- **Get more out of a small/cheap local model for free.** On a 3.4B model,
  switching from "just answer" to "write the full state each step" raised the
  solvable trace length from ~16 operations to 36+ (the model never broke before
  the test ran out of difficulty) — no training, just a prompt change. See
  `FINDINGS.md`.
- **Find the right scaffold, not just *a* scaffold.** The harness pits prompt
  formats against each other. It already surfaced a non-obvious result: tracking
  *only the queried object* (terse) silently reintroduces "ghost" errors on
  removals, while writing the *complete* state every step eliminates them. For
  this failure mode, completeness beats brevity — the opposite of the usual
  "minimal-edit scratchpad" advice.
- **Measure capability honestly.** Accuracy at one arbitrary difficulty is
  meaningless across models of different strength. The frontier metric
  auto-calibrates: it ramps difficulty until a strategy breaks, so a weak model
  and a strong model are compared at *their own* edges, and no compute is wasted
  on trivially-easy or hopelessly-hard problems.
- **Probe a specific, documented failure.** The tasks target the fragile
  global-suppression `REMOVE` mechanism from arXiv:2605.30233 — the case where
  models report a removed object as still present. You can dial up the stressors
  (removal density, confusable object names) to drive a model toward that
  failure and watch which prompt formats fix it.

## The research question behind it

From the synthesis page **`test-time-sampling-vs-retraining-ood`**: when you
force a model to externalize entity-state tracking into explicit tokens, does it
**route around** the fragile `REMOVE` mechanism, or does the failure persist?
That decides whether this class of failure is *recombination-OOD* (the
capability is latent and a test-time scaffold recovers it — no retraining) or
*capability-OOD* (the operation is absent and only retraining adds it). Result
so far: for below-ceiling models it is recombination-OOD — and the scaffold has
to externalize the specific thing that fails (the removal), or the fix doesn't
hold.

## How a task works

Each task is a sequence of `PUT` / `MOVE` / `REMOVE` operations over objects and
containers, with a deterministically tracked ground-truth final location and a
`"where is X?"` query. Datasets are balanced so ~half the queries target a
**removed** object (correct answer = `nowhere`) — the subset where the fragile
`REMOVE` tag is predicted to break.

All conditions end with the same machine-readable line so grading is identical
and the comparison is fair:

```
ANSWER: <container name, or the word "nowhere">
```

### Prompt conditions

| Condition | What it forces | Idea |
|--|--|--|
| `direct` | Answer immediately, no working out | parallel final-token aggregation (the native strategy) |
| `externalized` | Rewrite the **full** state of every container after each step | complete state externalized into tokens |
| `delta` | Track **only the queried object**, one line per step | minimal-edit externalization (Turing-Program style) |

### Error types (for wrong answers)

| Label | Meaning |
|--|--|
| `ghost` | Removed object reported as still in a container — the predicted suppression failure |
| `false_remove` | Present object reported as `nowhere` — over-suppression (the opposite leak) |
| `stale_or_wrong` | Named the wrong container — stale location / contamination |
| `no_answer` | No parseable answer and no salvageable completed trace |

## Quick start

Needs [uv](https://docs.astral.sh/uv/). For local models, [Ollama](https://ollama.com)
running with the model pulled.

```bash
uv run --extra dev pytest          # verify the deterministic core (no model needed)
ollama list                        # see which local models you have

# The headline experiment: how far does each prompt format extend the model?
uv run python frontier.py --backend ollama --model granite4.1:3b \
    --conditions direct,delta,externalized --ladder 8,12,16,24,36 \
    --n-tasks 20 --max-tokens 2048
```

Full command reference, including the reasoning-model and contamination
variants: **see `COMMANDS.md`**.

## What it can and can't tell you

This is a **behavioral** probe. It shows *whether* a prompt format helps, *how
far* it extends the frontier, and *which error type* it removes. It does **not**
prove the model is mechanistically using the suppression tag — that needs an
interpretability pass (activation patching / tag nullification on open weights).
The tasks are synthetic state-trackers: the right *kind* of probe (this is what
the length-generalization literature uses), but a frontier gain on this task
class is not an automatic transfer claim to real-world tasks.

## Layout

```
ete/
  tasks.py        task generation + deterministic ground truth (+ similar-name stressor)
  prompts.py      the three conditions: direct, externalized, delta
  backends.py     Ollama (/api/chat, with think toggle) + MiniMax (Anthropic endpoint)
  grading.py      answer extraction, trace salvage, error-type classification
  experiment.py   per-trial runner, aggregation, reporting
frontier.py       PRIMARY: competence-frontier sweep across difficulty (the headline metric)
sweep.py          fixed-ladder difficulty staircase (accuracy at each level)
twobytwo.py       reasoning {on,off} x prompt {direct,externalized} grid (reasoning models)
run.py            single fixed-difficulty run (one model, one config)
tests/test_core.py
FINDINGS.md       all results + interpretation
COMMANDS.md       runnable command reference
```

## Backends

- **Ollama** (`--backend ollama`, default) — local; never starts/stops the
  daemon. Supports a `think` toggle for reasoning models (deepseek-r1, qwen3).
- **MiniMax** (`--backend minimax`) — cloud, via the Anthropic-compatible
  Messages endpoint; needs `MINIMAX_API_KEY` in the environment. `thinking`
  blocks are stripped before grading.
