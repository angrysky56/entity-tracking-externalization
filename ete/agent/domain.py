"""Domain interface: what a tracking problem must provide to the engine.

A domain is defined by a *typed state schema* and a set of operations. The
generalized engine handles the discipline (full re-emission, completeness
checks, optional verification); the domain handles meaning.

State is deliberately a flat mapping slot -> list of entity ids, plus an
explicit NOWHERE slot. This single shape covers most entity-tracking problems:
  * objects in containers      : slot=container, entity=object
  * variables holding values   : slot=variable, entity=value (max 1 each)
  * files in directories       : slot=directory, entity=file
  * items in inventories        : slot=inventory, entity=item
The NOWHERE slot is what makes removal a *written* transition rather than an
erasure — the anti-ghost invariant, baked into the type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

NOWHERE = "nowhere"


@dataclass
class State:
    """Complete world-state: every slot mapped to its entities, plus NOWHERE.

    Invariant: every entity that has ever entered the world appears in exactly
    one slot's list (a real slot or NOWHERE). Entities are never deleted from
    the representation — removal moves them to NOWHERE. This is the type-level
    enforcement of the empirical anti-ghost finding.
    """

    slots: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.slots.setdefault(NOWHERE, [])

    def all_entities(self) -> set[str]:
        return {e for ents in self.slots.values() for e in ents}

    def location_of(self, entity: str) -> str:
        for slot, ents in self.slots.items():
            if entity in ents:
                return slot
        return NOWHERE

    def copy(self) -> "State":
        return State(slots={k: list(v) for k, v in self.slots.items()})

    def to_canonical(self) -> str:
        """Deterministic serialization for equality comparison (order-independent)."""
        return json.dumps(
            {k: sorted(v) for k, v in sorted(self.slots.items())},
            separators=(",", ":"),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self.to_canonical() == other.to_canonical()


@runtime_checkable
class Domain(Protocol):
    """What a concrete tracking problem must implement.

    The engine calls these; none of them know about LLMs. A domain is pure
    description + (optionally) a deterministic reducer used for verification and
    for harvesting correct training traces.
    """

    name: str

    def describe_rules(self) -> str:
        """One short paragraph of domain rules for the system prompt."""
        ...

    def slot_names(self) -> list[str]:
        """All slot names (real slots; NOWHERE is implicit)."""
        ...

    def render_operation(self, op: dict) -> str:
        """Render one operation as a natural-language instruction."""
        ...

    def initial_state(self) -> State:
        """Starting world-state (usually all slots empty)."""
        ...

    # --- optional but recommended: enables verification + trace harvesting ---

    def reduce(self, state: State, op: dict) -> State:
        """Deterministic ground-truth transition. Raise NotImplementedError if
        the domain has no closed-form reducer (then the agent runs unverified).
        """
        ...

    def parse_state(self, text: str) -> State | None:
        """Parse the model's emitted state line back into a State, or None if it
        cannot be parsed (used for completeness/agreement checks).
        """
        ...
