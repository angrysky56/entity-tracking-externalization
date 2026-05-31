"""Concrete domains, to demonstrate the engine is domain-independent.

Each maps a different real problem onto the same slot->entities State:
  * ContainerDomain: objects in containers (the original entity-tracking task).
  * VariableDomain:  variables holding values (slot=variable, <=1 entity each);
                     a structurally different problem the SAME engine tracks.

Adding a domain means implementing the Domain protocol — no engine changes.
"""

from __future__ import annotations

import re

from .domain import NOWHERE, State


def _parse_state_generic(text: str) -> State | None:
    """Parse the last 'STATE: a=x,y | b= | nowhere=z' line into a State."""
    lines = [l.strip() for l in (text or "").splitlines() if "STATE:" in l]
    if not lines:
        return None
    body = lines[-1].split("STATE:", 1)[1].strip()
    slots: dict[str, list[str]] = {}
    for chunk in body.split("|"):
        if "=" not in chunk:
            continue
        slot, items = chunk.split("=", 1)
        slot = slot.strip()
        ents = [e.strip() for e in items.split(",") if e.strip()]
        if slot:
            slots[slot] = ents
    if not slots:
        return None
    slots.setdefault(NOWHERE, [])
    return State(slots=slots)


class ContainerDomain:
    """Objects move between containers; 'take out' sends an object to nowhere."""

    def __init__(self, containers: list[str]) -> None:
        self.name = "containers"
        self._containers = list(containers)

    def describe_rules(self) -> str:
        return ("There are containers and objects. Each object is in at most one "
                "container. Removing an object sends it to 'nowhere'.")

    def slot_names(self) -> list[str]:
        return list(self._containers)

    def initial_state(self) -> State:
        return State(slots={c: [] for c in self._containers} | {NOWHERE: []})

    def render_operation(self, op: dict) -> str:
        k, obj = op["kind"], op["obj"]
        if k == "PUT":
            return f"Put the {obj} in the {op['container']}."
        if k == "MOVE":
            return f"Move the {obj} to the {op['container']}."
        return f"Take the {obj} out."

    def reduce(self, state: State, op: dict) -> State:
        if "__probe__" in op:  # reducer-existence probe
            return state
        s = state.copy()
        obj = op["obj"]
        for ents in s.slots.values():
            if obj in ents:
                ents.remove(obj)
        dest = NOWHERE if op["kind"] == "REMOVE" else op["container"]
        s.slots.setdefault(dest, []).append(obj)
        return s

    def parse_state(self, text: str) -> State | None:
        return _parse_state_generic(text)


class VariableDomain:
    """Variables hold a single value; 'clear x' sends x's value to nowhere.

    Structurally different from containers (slots hold <=1 entity, entities are
    values) yet tracked by the identical engine — the generalization proof.
    """

    def __init__(self, variables: list[str]) -> None:
        self.name = "variables"
        self._vars = list(variables)

    def describe_rules(self) -> str:
        return ("There are variables. Each variable holds at most one value. "
                "Assigning overwrites; clearing a variable sends its value to "
                "'nowhere'.")

    def slot_names(self) -> list[str]:
        return list(self._vars)

    def initial_state(self) -> State:
        return State(slots={v: [] for v in self._vars} | {NOWHERE: []})

    def render_operation(self, op: dict) -> str:
        if op["kind"] == "SET":
            return f"Set {op['var']} = {op['value']}."
        return f"Clear {op['var']}."

    def reduce(self, state: State, op: dict) -> State:
        if "__probe__" in op:
            return state
        s = state.copy()
        var = op["var"]
        old = list(s.slots.get(var, []))
        s.slots[var] = []
        for v in old:  # displaced values go to nowhere (still tracked, not erased)
            s.slots.setdefault(NOWHERE, []).append(v)
        if op["kind"] == "SET":
            s.slots[var] = [op["value"]]
        return s

    def parse_state(self, text: str) -> State | None:
        return _parse_state_generic(text)
