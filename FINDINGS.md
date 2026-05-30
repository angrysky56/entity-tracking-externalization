# Findings

## Reasoning-toggle 2x2 — deepseek-r1:latest (in progress, 2026-05-30)

**Design.** The real test of "can explicit externalization substitute for
free-form reasoning, and could we switch reasoning off when it triggers?"
deepseek-r1 (8.2B qwen3 distill) honors Ollama's `think: true|false`, unlike
MiniMax-M2.7. Four cells on one task set: reasoning {on, off} x prompt {direct,
externalized}. Runner: `twobytwo.py`. The toggle was verified working —
`think=True` returns a separated `thinking` field + clean content;
`think=False` returns content with zero thinking.

**Decisive comparison:** `reason_off + externalized` (the proposal: scaffold
replaces reasoning) vs `reason_on + direct` (native R1). If the former matches
the latter, structured externalization can *replace* reasoning on this task
class — switch reasoning off, keep accuracy, save the reasoning tokens.

**Cost reality (measured).** A single reason-on trial at n_ops=16 took **~150 s**
on the RTX 3060 (eval_count≈557, `done_reason=stop` — a clean ~1900-char trace,
**not** a runaway loop; just slow). That makes large reason-on runs impractical
locally: 40 trials/cell x 2 reasoning-on cells ≈ 3+ hours. Pilot runs use small
n_tasks. This is itself a finding: **R1-style reasoning is ~100x the wall-clock
cost of a direct answer**, so if the scaffold can replace it, the token/latency
savings are large — which is the whole point of the "switch reasoning off"
strategy.

### Pilot result — n=6 only, UNDERPOWERED, direction-hint not a conclusion

deepseek-r1:latest, n_tasks=6 (12 trials/cell), n_ops=12, remove_prob=0.45,
temp=0, cap 3072/1536. At n=6 a single task flip = 0.17 and a "0.33 ghost rate"
is 2 of 6 items — treat everything below as a hint to be confirmed at n>=30.

| | direct | externalized |
|--|--|--|
| reason_on | 0.67 | 0.67 |
| reason_off | **0.83** | 0.67 |

Hints, all tentative:

1. **Reasoning did not help and slightly hurt** (reason_on direct 0.67 <
   reason_off direct 0.83). Both reason_on cells had 2 `no_answer` — R1's trace
   sometimes ran long and never emitted a clean ANSWER line (grading/cap
   artifact, partly).
2. **Externalization HURT on R1** — opposite of granite. reason_off direct 0.83
   → externalized 0.67, and the externalized cell produced **2 ghost errors**
   (the REMOVE-failure signature) that the direct cell did not. So on this more
   capable model the rigid external-state format appears to *interfere* rather
   than scaffold — consistent with the "don't externalize above the ceiling"
   lesson, but now sharper: externalization is not free, it can inject errors.
3. The "CAN replace reasoning" flag fired only because native reasoning was
   weak (0.67), not because the scaffold was strong.

**Caveats / fixes applied.** The original n=20 run was abandoned: R1 reasoning is
~150 s/trial, so 160 trials ≈ 3+ h locally. The 2x2 runner was patched to save
raw transcripts (it previously saved only summaries, so these ghost errors could
not be audited — a real gap). Next run needs n>=30 and transcript inspection to
tell real tracking ghosts from format artifacts before any of the above is
trusted.

---

## Difficulty staircase — granite4.1:3b (2026-05-30): externalization is a capability *extender*

The decisive run. Both conditions swept across trace length (n_ops), objects/
containers fixed, remove_prob=0.45, 20 tasks/level. `sweep.py`.

| n_ops | direct | externalized | gain | direct ghost | ext ghost |
|------|--------|--------------|------|------|------|
| 8 | 0.75 | 0.90 | +0.15 | 0.00 | 0.05 |
| 16 | 0.65 | 0.85 | +0.20 | 0.00 | 0.05 |
| 24 | 0.40 | 0.80 | **+0.40** | 0.00 | 0.05 |
| 32 | 0.40 | 0.90 | **+0.50** | 0.00 | 0.00 |
| 40 | *(slow; last level did not complete in budget — externalized outputs hit the verbose ceiling on a 3B local model)* |

**The gain widens monotonically with difficulty.** `direct` falls off a cliff
(0.75 → 0.40) as the trace lengthens and working-memory pressure mounts;
`externalized` stays flat at 0.80–0.90. So externalization is **not a fixed
offset** — it extends the *range of tasks the model can do at all*. At n_ops=32
the model with externalization (0.90) is succeeding on tasks where its direct
mode (0.40) has essentially failed.

This is the core mechanism made visible: writing state to tokens converts a
held-in-activations computation (which degrades with sequence length) into a
sequence of fresh, local sub-problems (which doesn't). The harder the task
relative to internal capacity, the bigger the rescue.

### Note: small externalized ghosts at easy levels

A new wrinkle — at n_ops 8/16/24 the *externalized* condition showed a small
ghost rate (~0.05) while direct showed none. Worth a look: at easy difficulty
direct answering is reliable, but the verbose step-by-step format occasionally
introduces a transcription slip (carrying a removed object forward in the
written state). The format has its own low-rate failure mode that only matters
when the task is easy enough that direct wasn't going to fail anyway — i.e. it
costs a little exactly where it helps least, consistent with the "don't
externalize below the ceiling" rule.

## Reasoning models: can the lever apply, and can reasoning be switched off?

**Empirical finding (2026-05-30): MiniMax-M2.7 thinking cannot be disabled via
the API.** A direct call with `thinking: {"type": "disabled"}` still returned a
`thinking` block (block types: `['thinking', 'text']`). This matches the
upstream MiniMax-M2 issue tracker: the model does not honor thinking-disable.
So the "shut reasoning off when externalization triggers" strategy is **not
testable on M2.7** and likely not implementable on it at all.

Consequences for the strategy:

- On a model where reasoning *can* be toggled (many Ollama thinking models,
  DeepSeek, Qwen3 with `/think` off, etc.), the head-to-head is runnable:
  native-reasoning vs structured-externalization-with-reasoning-off, on a task
  hard enough that native reasoning fails. That is the real test of "does the
  scaffold beat free-form reasoning, and can we drop the expensive reasoning."
- On M2.7 specifically, the only available comparison is native-reasoning vs
  externalization-prompt-with-reasoning-still-on — which confounds the two and
  is why M2.7 ceilinged earlier (the thinking channel already externalizes).

| Model | Type | `direct` acc | `externalized` acc | Read |
|--|--|--|--|--|
| MiniMax-M2.7 | cloud reasoning | 1.00 | 0.71 (all artifacts) | ceiling — hypothesis not testable |
| granite4.1:3b (n=14) | local instruct, non-reasoning | 0.71 | **1.00** | below ceiling — externalization repairs |
| granite4.1:3b (n=60) | local instruct, non-reasoning | 0.67 | **0.88** | confirmed: +22pp, ghosts eliminated |

The two runs are complementary: M2.7 establishes that a strong reasoning model
has no fragility to expose here, and granite — small, non-reasoning, below
ceiling — is where the manipulation actually does work.

## Pilot 2 — granite4.1:3b (2026-05-30)

**Model:** granite4.1:3b (IBM Granite, 3.4B Q4_K_M, capabilities: completion +
tools; **no thinking/reasoning channel** — confirmed via `ollama show`).
**Config:** 14 tasks, `--n-ops 12 --n-objects 5 --remove-prob 0.45 --max-tokens 1536 --temperature 0`, local Ollama.

| Condition | Overall | Removed-query | Present-query | Ghost rate |
|--|--|--|--|--|
| `direct` | 0.71 | 0.71 | 0.71 | 0.00 |
| `externalized` | **1.00** | 1.00 | 1.00 | 0.00 |

**Headline: +0.29 gain, identical on removed- and present-query items.**

Direct-condition errors (4 of 14): 2 `no_answer` (no clean ANSWER line under the
terse prompt), 1 `stale_or_wrong` (wrong container), 1 `false_remove` (said
"nowhere" for a present object). Externalized condition: **zero errors of any
kind, zero artifacts.**

### Read against the decision rule

The synthesis page's rule: a *disproportionate* gain on removed-query items +
a ghost-rate drop ⇒ externalization routes around the fragile REMOVE tag
(recombination-OOD). A *uniform* gain across subsets ⇒ the failure is broader
than the REMOVE tag.

Here the gain is **uniform** (+0.29 on both subsets) and there were **no ghost
errors to drop**. So:

- **This is recombination-OOD, not capability-OOD.** Externalization fully
  repaired the failures with no retraining — the correct computation was latent
  in the model and surfacing it into tokens recovered it. This is the "test-time
  fix, no retraining" side of the boundary, confirmed behaviorally.
- **But what it repaired is general state-tracking, not specifically the
  REMOVE-suppression mechanism.** The direct errors were a stale location and an
  over-suppression, not the ghost signature (removed-object-reported-present)
  that the entity-tracking paper predicts. At this difficulty granite's REMOVE
  handling is fine; its generic multi-step tracking is what externalization
  fixed.

### Caveats

- n=14 is a pilot. The 2 direct `no_answer`s are format misses, arguably not
  tracking failures; they modestly inflate the measured gain. A larger run
  (n≥60) and a stricter direct prompt that guarantees an ANSWER line would
  tighten the estimate.
- To actually exercise the *REMOVE-specific* hypothesis, push difficulty until
  ghost errors appear in the direct condition (more accumulated removes, similar
  object names for contamination), then see whether externalization removes the
  ghosts specifically.

## Confirmation — granite4.1:3b, n=60 (2026-05-30)

Same config, 60 tasks (120 trials). Confirms the pilot past noise **and** the
larger sample surfaces the REMOVE-specific signature the n=14 run was too small
to show.

| Condition | Overall | Removed-query | Present-query | Ghost rate |
|--|--|--|--|--|
| `direct` | 0.667 | 0.633 | 0.700 | **0.067** (4 cases) |
| `externalized` | **0.883** | 0.833 | 0.933 | **0.000** |

**Gain: +0.20 removed, +0.23 present (~+22pp overall).**

Direct errors (20 of 60): 7 `false_remove`, 6 `no_answer`, 4 `ghost`, 3
`stale_or_wrong`. Externalized errors (7 of 60): 4 `stale_or_wrong`, 3
`false_remove` — **zero ghosts, zero no_answer**.

### What the larger sample adds

1. **The fragile-REMOVE signature appeared and was eliminated.** At n=60 the
   direct condition produced 4 `ghost` errors — a removed object reported as
   still present, the exact failure the entity-tracking paper predicts. The
   externalized condition produced **zero ghosts**. This is the first run to
   actually touch the REMOVE-tag hypothesis (not just general tracking), and
   externalization removed the signature cleanly.
2. **The `no_answer` artifact is purely a direct-prompt problem** (6 → 0). The
   step-by-step structure forces a gradeable answer line.
3. **Broad + specific repair.** The gain is roughly uniform across query types
   (general multi-step tracking improves) *and* the ghost subclass is wiped out
   (the REMOVE mechanism specifically improves). Both, not either/or.

### Bottom line

For a 3.4B non-reasoning local model, prompting it to externalize entity state
step-by-step gives a **~22-point accuracy gain on structured tracking and
eliminates fragile-REMOVE ghost errors, with no fine-tuning.** This is
recombination-OOD confirmed at scale: the capability was latent; surfacing the
computation into tokens recovers it. A genuine, free win for cheap local models
on structured-state tasks.

### Scaling conjecture (untested)

The mechanism is governed by the gap between task demand and the model's
*internal* capacity, not by model size directly. Predictions:

- Bigger models get the same repair, but only on **harder tasks** (longer
  traces, more entities, denser removes) that put *them* near their ceiling.
- On tasks well below a model's ceiling the gain → 0 and can go negative
  (wasted tokens, artifacts) — this is the MiniMax-M2.7 ceiling result.
- **Reasoning models benefit least** from explicit externalization: their
  thinking channel already externalizes internally, so explicit CoT is partly
  redundant. The cleanest wins are small/non-reasoning models near their ceiling
  — exactly this granite case.
- Capability-OOD (operation absent from the model) is unfixable by
  externalization at any size; that needs retraining.


---

## Pilot 1 — MiniMax-M2.7 (2026-05-29): the ceiling result


**Model:** MiniMax-M2.7 (Anthropic-compatible cloud endpoint)
**Config:** 14 tasks, `--n-ops 12 --n-objects 5 --remove-prob 0.45 --temperature 0`
**Replication:** run twice; results identical (deterministic at temp 0).

| Condition | Overall | Removed-query | Present-query | Ghost rate |
|--|--|--|--|--|
| `direct` | **1.00** | 1.00 | 1.00 | 0.00 |
| `externalized` | 0.71 | 0.57 | 0.86 | 0.00 |

### What actually happened

Every `externalized` "error" was an **instrumentation artifact**, not a wrong
answer:

- 2 cases: empty output — the reply was entirely a `thinking` block with no
  `text` block, so nothing was graded.
- 1 case: token-budget truncation — the step-by-step trace ran past `max_tokens`
  and never reached the final `ANSWER:` line.

**Zero wrong locations. Zero ghost errors in either condition.** The model never
mis-tracked a REMOVE; it sometimes failed to *emit* a gradeable answer under the
externalized prompt.

## Interpretation

The original hypothesis (from the synthesis page
`test-time-sampling-vs-retraining-ood`) was: does forcing entity tracking into
explicit CoT **route around** the fragile global-suppression `REMOVE` mechanism
documented in arXiv:2605.30233?

For MiniMax-M2.7 the result **inverts the question**:

1. **There is no fragility to route around at this difficulty.** The fragile
   `REMOVE` tag is a property of the *specific (smaller/older) models* in the
   entity-tracking paper, not a universal of transformers. A current reasoning
   model tracks PUT/MOVE/REMOVE perfectly here — plausibly because its
   `thinking` channel already does the step-by-step state update internally.

2. **Forced externalization can hurt.** It consumed the token budget and pushed
   content into the reasoning channel, manufacturing artifacts where direct
   answering was clean.

3. **This is a difficulty/capability floor, not a test of the boundary.** The
   recombination-vs-capability OOD boundary only becomes testable in a regime
   where `direct` itself fails. M2.7 at n_ops=12 is far below that.

## Connection: same lesson as the MOPS drift falsification

The wiki page `cross-layer-drift-falsification` reaches an analogous conclusion
from the mechanistic side: a geometric apparatus (sheaf drift over the residual
stream) failed to detect a falsity signal that actually lives in a *sparse,
directional* substrate — the wrong observable for where the signal lives.

This experiment is the behavioral mirror. A **behavioral output-accuracy probe**
cannot see a sub-threshold internal mechanism. When the competent computation
lives in the substrate (here, the thinking channel), a surface measure (final
answer accuracy) reports a ceiling and reveals nothing about the mechanism — and
the externalization harness both misses the internal computation and interferes
with it.

General form: *if you want to test for a mechanism, your observable has to be
able to see that mechanism.* Output accuracy can't.

## What would make this a real test of the hypothesis

1. **Break `direct` first.** Find the model/difficulty regime where the direct
   condition's removed-query accuracy drops below ~0.8 — either a weaker model
   (local `gemma4`, or a small instruct model with no reasoning channel) or a
   much harder task (longer traces, more objects, denser removes, distractor
   re-placements). Only then can externalization show a repair effect.

2. **Use an instruct (non-reasoning) model.** A model without a private thinking
   channel cannot externalize internally, so the direct-vs-externalized contrast
   is clean. This is the better substrate for the *precise* REMOVE-tag question.

3. **Go mechanistic for the real claim.** A behavioral probe can show *whether*
   externalization helps; it cannot prove the model uses the suppression tag.
   That needs activation patching / tag nullification on an open-weights model
   (the kind of substrate-level probe the drift page argues for).

## Fix applied after this run

The MiniMax backend now falls back to the `thinking` block when no `text` block
is returned, so the "empty output" artifact no longer produces spurious
`no_answer` grades. Truncation is mitigated by the `--max-tokens` cap (raise it
for the externalized condition, which is verbose by construction).
