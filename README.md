# Entity-Tracking Externalization Probe

A small, self-contained experiment testing one question from the synthesis page
**`test-time-sampling-vs-retraining-ood`**:

> When you force an LM to externalize entity-state tracking into an explicit
> chain of thought, does it **route around** the fragile global-suppression
> `REMOVE` mechanism (arXiv:2605.30233), or does the failure **persist**?

The answer decides whether this class of failure is *recombination-OOD* (fixable
at test time, no retraining) or *capability-OOD* (needs retraining).

## The design

Each task is a sequence of `PUT` / `MOVE` / `REMOVE` operations over objects and
containers, with a deterministically tracked ground-truth final location and a
`"where is X?"` query. The dataset is balanced so ~half the queries target a
**removed** object (correct answer = `nowhere`) — that subset is where the
fragile `REMOVE` tag is predicted to break.

Two conditions, graded identically (last `ANSWER:` line):

| Condition | What it forces | Maps to |
|--|--|--|
| `direct` | Answer immediately, no reasoning | parallel final-token aggregation |
| `externalized` | Rewrite full container state after every step, then answer | sequential state externalized into tokens |

## What to look for

The runner prints the decisive contrast:

- **Gain on removed-query items vs present-query items.** A *disproportionate*
  externalization gain on removed-query items, plus a **ghost-rate drop**, means
  externalization routes around the REMOVE tag → recombination-OOD → test-time
  fix, no retraining.
- **Similar gains across subsets** → the failure is preserved → points to
  capability-OOD.

`ghost` = a removed object reported as still in a container (the predicted
suppression-failure signature). `false_remove` = the opposite over-suppression.

## Running it

Two backends: local **Ollama** (`--backend ollama`, default) and cloud
**MiniMax** via the Anthropic-compatible endpoint (`--backend minimax`, needs
`MINIMAX_API_KEY` in the environment). Both need [uv](https://docs.astral.sh/uv/);
Ollama also needs the daemon running with the model pulled.

```bash
ollama list                     # see what you have pulled (Ollama backend)
uv run --extra dev pytest       # verify the deterministic core (no model needed)

# Local (Ollama). gemma4 is slow on a 12 GB card for the verbose externalized
# condition — keep n-tasks small or expect ~40+ min for 60.
uv run python run.py --backend ollama --model gemma4:latest --n-tasks 14 --max-tokens 1024
uv run python run.py --backend ollama --model qwen3:0.6b --n-tasks 6 --no-balance  # smoke test

# Cloud (MiniMax, off your GPU, fast). Reasoning model — set max-tokens high for
# the externalized condition or it truncates mid-trace.
uv run python run.py --backend minimax --model MiniMax-M2.7 \
    --n-tasks 14 --n-ops 12 --n-objects 5 --remove-prob 0.45 --max-tokens 4096
```

Useful knobs: `--n-ops` (trace length / difficulty), `--remove-prob` (REMOVE
density), `--max-tokens` (generation cap), `--temperature`, `--seed`. Raw
responses + summary are written to `results/` as timestamped JSON.

> **Result so far (see `FINDINGS.md`):** on MiniMax-M2.7, `direct` hits 100%
> with zero ghost errors and externalization only *adds* artifacts — the fragile
> REMOVE failure doesn't appear at this difficulty, so the hypothesis isn't yet
> testable on this model. The informative regime is one where `direct` itself
> fails (weaker / non-reasoning model, or harder tasks).

## Scope and honesty

This is a **behavioral** probe, not the mechanistic intervention. It can show
*whether* externalization helps and *which error type* it removes; it cannot, on
its own, prove the model is using the specific suppression tag. A natural
follow-on is a mechanistic pass (e.g. activation patching / tag nullification on
an open-weights model) to confirm the mechanism — and a sampling pass
(entropy-cut / self-consistency over the externalized trace) to test the
test-time-repair claim directly.

## Layout

```
ete/
  tasks.py        task generation + deterministic ground truth
  prompts.py      the two conditions
  backends.py     Ollama (/api/chat) + MiniMax (Anthropic endpoint); never starts/stops a server
  grading.py      answer extraction + error-type classification
  experiment.py   runner, aggregation, reporting
run.py            CLI entrypoint
tests/test_core.py
FINDINGS.md       pilot results + interpretation
```
