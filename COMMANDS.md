# Commands

Runnable reference for every script in the harness. All commands assume:

```bash
cd ~/Repositories/ai_workspace/entity-tracking-externalization
```

and that [uv](https://docs.astral.sh/uv/) is installed. Local (Ollama) runs need
the daemon running (`ollama serve`) and the model pulled (`ollama pull <model>`,
check with `ollama list`). The harness never starts or stops the Ollama server.

Results are written to `results/` as timestamped JSON. The deterministic core
needs no model:

```bash
uv run --extra dev pytest          # 8 tests, ~instant
```

---

## 1. `frontier.py` — the primary experiment

Measures the **competence frontier**: the hardest trace length each prompt
format still solves at/above the pass threshold, and the **frontier shift** vs
`direct` (extra steps unlocked). Ramps difficulty per condition and stops a
condition once it breaks, so no compute is wasted above each strategy's ceiling.

**Main run (local, non-reasoning model):**

```bash
uv run python frontier.py --backend ollama --model granite4.1:3b \
    --conditions direct,delta,externalized --ladder 8,12,16,24,36 \
    --n-tasks 20 --max-tokens 2048
```

**Contamination stressor** — confusable object names ("widget 1", "widget 2", …)
to drive cross-object contamination and test whether full-state externalization
still suppresses ghosts when objects are hard to tell apart:

```bash
uv run python frontier.py --backend ollama --model granite4.1:3b \
    --conditions direct,delta,externalized --ladder 8,12,16,24,36 \
    --n-tasks 20 --max-tokens 2048 --similar-names
```

**Reasoning model, the substitution question** — does the externalization
scaffold reach the frontier of native reasoning *without* paying for reasoning?
Run the same model twice and compare frontiers:

```bash
# reasoning OFF + scaffold (the proposal)
uv run python frontier.py --backend ollama --model deepseek-r1:latest \
    --conditions direct,externalized --ladder 8,12,16,24 \
    --n-tasks 12 --max-tokens 4096 --think off

# reasoning ON, direct (native baseline) — SLOW: R1 ~150 s/trial, run overnight
uv run python frontier.py --backend ollama --model deepseek-r1:latest \
    --conditions direct --ladder 8,12,16,24 \
    --n-tasks 12 --max-tokens 4096 --think on
```

If `reason_off + externalized` reaches a frontier >= `reason_on + direct`, the
scaffold can replace reasoning on this task class — the "switch reasoning off
when this triggers" idea, validated, at ~1/100th the latency.

**Watch the model think** — `--verbose` echoes every full response (reasoning
trace and/or state rewrite) as it streams:

```bash
uv run python frontier.py --backend ollama --model granite4.1:3b \
    --conditions direct,externalized --ladder 8,16 --n-tasks 6 \
    --max-tokens 2048 --verbose
```

**Flags:** `--conditions` (subset of `direct,delta,externalized`), `--ladder`
(ascending n_ops rungs), `--n-tasks` (trials/rung), `--pass-threshold` (default
0.6), `--remove-prob`, `--n-objects`, `--n-containers`, `--similar-names`,
`--think {on,off,default}`, `--temperature`, `--max-tokens`, `--seed`,
`--verbose`.

---

## 2. `sweep.py` — fixed difficulty staircase

Runs both `direct` and `externalized` at every rung of a fixed ladder and prints
accuracy + ghost rate at each level (no early stop). Use when you want the full
accuracy curve across difficulty rather than just the crossing point.

```bash
uv run python sweep.py --backend ollama --model granite4.1:3b \
    --n-tasks 20 --ladder 8,16,24,32,40 --remove-prob 0.45 --max-tokens 2048
```

**Flags:** `--ladder`, `--n-tasks` (per level), `--remove-prob`, `--n-objects`,
`--n-containers`, `--temperature`, `--max-tokens`, `--seed`.

---

## 3. `twobytwo.py` — reasoning x prompt grid (reasoning models)

Crosses reasoning {on, off} with prompt {direct, externalized} on one shared
task set at a single difficulty. Saves raw transcripts so ghost errors can be
audited. Use for a focused look at all four cells at one difficulty; use
`frontier.py --think` for the capability-range version.

```bash
uv run python twobytwo.py --model deepseek-r1:latest \
    --n-tasks 30 --n-ops 16 --remove-prob 0.45 --max-tokens 4096 --verbose
```

> **Cost warning:** reasoning-on trials are ~150 s each on a 12 GB GPU. 30 tasks
> = 120 trials, ~3 h. Start with `--n-tasks 6` for a quick read; run large jobs
> overnight, backgrounded. The two `reason_on` cells run first (the slow ones).

**Flags:** `--n-tasks`, `--n-ops`, `--remove-prob`, `--n-objects`,
`--n-containers`, `--temperature`, `--max-tokens`, `--seed`, `--verbose`.

---

## 4. `run.py` — single fixed-difficulty run

One model, one configuration, both `direct` and `externalized`. The simplest
entry point; good for a smoke test or a quick single data point.

```bash
# local smoke test
uv run python run.py --backend ollama --model granite4.1:3b \
    --n-tasks 14 --n-ops 12 --remove-prob 0.45 --max-tokens 1536

# MiniMax cloud (needs MINIMAX_API_KEY in the environment)
uv run python run.py --backend minimax --model MiniMax-M2.7 \
    --n-tasks 14 --n-ops 12 --n-objects 5 --remove-prob 0.45 --max-tokens 4096
```

**Flags:** `--n-ops`, `--n-tasks`, `--remove-prob`, `--n-objects`,
`--n-containers`, `--no-balance` (don't force ~50% removed-query tasks),
`--temperature`, `--max-tokens`, `--seed`.

---

## Long / slow runs

A tool call that runs past ~4 minutes can time out in some shells even though
the process keeps going. For long jobs (reasoning models, big sweeps), background
to a log and poll it:

```bash
nohup uv run python frontier.py --backend ollama --model deepseek-r1:latest \
    --conditions direct,externalized --ladder 8,12,16,24 \
    --n-tasks 12 --max-tokens 4096 --think off > frontier_r1.log 2>&1 &

tail -f frontier_r1.log        # watch progress; Ctrl-C just stops watching
```

The MiniMax key lives in `~/.bashrc`, so a backgrounded job needs a login/
interactive shell to see it (`bash -ic '...'`) — or export it in the current
shell first.

## Reading results

Every run prints a summary and saves JSON to `results/`:

- `frontier-<model>-<stamp>.json` — `frontier` (n_ops per condition) + `curve`
  (accuracy/ghost/errors at each rung).
- `sweep-<model>-<stamp>.json` — per-level summaries.
- `twobytwo-<model>-<stamp>.json` — four cells + raw per-trial `records`.
- `results-<model>-<stamp>.json` — single-run summary + records.

Common knobs across all runners:

| Flag | Effect |
|--|--|
| `--n-ops` / `--ladder` | trace length = difficulty |
| `--remove-prob` | density of REMOVE ops (raises ghost pressure) |
| `--similar-names` | confusable objects (frontier.py only) |
| `--n-objects` / `--n-containers` | state-space size |
| `--max-tokens` | generation cap — raise for verbose/externalized + reasoning |
| `--temperature` | 0 = deterministic (note: deepseek-r1 ships temp 0.6 by default; the harness overrides to your value) |
| `--seed` | dataset reproducibility |
| `--think` | reasoning toggle, reasoning models (frontier.py) |
| `--verbose` | echo full model output per trial (frontier.py, twobytwo.py) |
