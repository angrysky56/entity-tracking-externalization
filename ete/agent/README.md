# Generalized full-state tracking agent

`ete/agent/` — the deployable, domain-independent version of the finding in
`FINDINGS.md`.

## The one invariant it generalizes

The experiments showed the fix for silent-suppression ("ghost") errors is not
"show your work" but **maintain a complete, typed world-state and re-emit ALL of
it after every operation — removed entities tracked explicitly in a `nowhere`
slot, never erased.** Tracking only the queried slot (`delta`) reintroduced the
ghost. This module bakes that invariant into the type system and the step loop,
independent of any specific problem.

## Architecture: a clean seam between model and code

```
operations ─▶ TrackingAgent (engine, domain-independent)
                 │  per step: prompt model for COMPLETE state
                 │            parse → completeness check → repair if shrunk
                 │            (optional) verify vs domain.reduce() ground truth
                 ▼
              State (typed: slot → [entities], + explicit `nowhere`)
                 ▲
              Domain (problem-specific: rules, ops, reducer, parser)
```

- **State** (`domain.py`): flat `slot -> [entities]` plus a mandatory `nowhere`
  slot. Removal = move to `nowhere`, so the representation can never silently
  drop an entity. The anti-ghost rule is a type invariant, not a prompt plea.
- **TrackingAgent** (`core.py`): the universal engine. Forces full re-emission
  each step; the _completeness check_ (no entity from the prior step missing) is
  the runtime anti-ghost guard; `repair_incomplete` re-adds any dropped entity
  to `nowhere` rather than letting it vanish.
- **Domain** (`domain.py` Protocol): what a problem supplies — rules text, slot
  names, operation rendering, an optional deterministic `reduce()` and a
  `parse_state()`. No domain code knows about LLMs.

## Two modes

- `verify=False` (deployment): trust the model's state but enforce completeness.
- `verify=True` (eval / data harvesting): compare every step to `reduce()`'s
  ground truth. A trace is `clean` only if every step parsed, stayed complete,
  and matched truth. **Clean traces are the fine-tuning corpus** — this is the
  recursive-self-training loop: agent prompt → filtered correct traces → SFT.

## Generalization, demonstrated

`domains.py` ships two structurally different problems tracked by the _same
engine_:

- `ContainerDomain` — objects in containers (the original task).
- `VariableDomain` — variables holding values (slots hold ≤1 entity, entities
  are values). Different semantics, zero engine changes.

Adding a domain (files/dirs, balances, graph nodes, inventory) means
implementing the `Domain` protocol — nothing in `core.py` changes.

## Run it

```bash
uv run --extra dev pytest tests/test_agent.py     # logic (scripted backend, no model)
uv run python -m ete.agent.smoke --model granite4.1:3b   # live, both domains
```

## Scope

The completeness guard makes ghosts _structurally impossible to hide_, but
schema enforcement constrains shape, not truth — a wrong transition is still
wrong. That is why `verify=True` + a deterministic `reduce()` matters: it is the
only thing that certifies a transition correct. For domains with no closed-form
reducer, the agent still runs (completeness-enforced) but traces are unverified.

## Wiki

Full project analysis, findings, and cross-links →
[entity-tracking-externalization](https://github.com/angrysky56/LLM-WIKI/wiki/entities/projects/entity-tracking-externalization)
