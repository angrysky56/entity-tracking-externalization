"""Generalized full-state tracking agent.

The empirical finding this generalizes (see ../../FINDINGS.md): silent-suppression
failures (an entity that should be gone reported as still present) are only
caught by forcing the model to write the COMPLETE state after every operation.
Tracking only the queried slot (delta) reintroduces the ghost. So the universal
invariant is not "show your work" but:

    maintain an explicit, complete, typed world-state; re-emit ALL of it after
    every operation; never drop an entity — removal is a written transition, not
    an erasure.

What generalizes (the engine, domain-independent):
  * a typed State the agent must fully serialize each step,
  * a step loop that re-emits and validates completeness,
  * optional verification of each transition against a deterministic reducer.

What a domain supplies (the Domain interface in domain.py):
  * the state schema (what slots/entities exist),
  * how to render an operation and the state as text,
  * optionally a reducer: (state, op) -> state for ground-truth checking and
    for harvesting verified training traces.

Entity-tracking (boxes/objects) is ONE domain; variables/values, files/dirs,
graph nodes/edges, balances, inventory are others — same engine.
"""

from __future__ import annotations

__all__ = ["Domain", "State", "TrackingAgent", "StepTrace", "AgentResult"]

from .domain import Domain, State
from .core import TrackingAgent, StepTrace, AgentResult
