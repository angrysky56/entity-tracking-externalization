"""The generalized tracking engine — domain-independent.

Per step: ask the model to emit the COMPLETE updated state, parse it, check it
for completeness (no dropped entities), and — if the domain has a reducer —
compare against ground truth. The engine never lets the representation shrink:
the completeness check is the runtime form of the anti-ghost invariant.

Two verification modes:
  * verify=False (deployment): trust the model's emitted state, but still
    enforce completeness (reject/repair states that dropped entities).
  * verify=True (harvesting/eval): compare each emitted state to the reducer's
    ground truth; a trace is "clean" only if every step matched. Clean traces
    are exactly what you fine-tune on (recursive self-training).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .domain import NOWHERE, Domain, State


@dataclass
class StepTrace:
    """Record of one operation step."""

    index: int
    operation_text: str
    raw_model_output: str
    parsed_state: State | None
    complete: bool                 # no entities dropped vs previous step
    matches_truth: bool | None     # vs reducer, or None if no reducer
    truth_state: State | None


@dataclass
class AgentResult:
    """Outcome of running the agent over a full operation sequence."""

    final_state: State | None
    steps: list[StepTrace] = field(default_factory=list)
    clean: bool = False            # every step parsed, complete, and (if checked) matched truth
    query: str | None = None
    predicted_location: str | None = None
    truth_location: str | None = None

    @property
    def correct(self) -> bool | None:
        if self.predicted_location is None or self.truth_location is None:
            return None
        return self.predicted_location == self.truth_location


_SLOT_LINE_HINT = (
    "After EACH step, output one line beginning 'STATE:' followed by the COMPLETE "
    "state — every slot and its contents, including empty slots and the special "
    "'nowhere' slot for anything removed. Never omit a slot. Never drop an item "
    "you have seen; if it is removed, move it to 'nowhere'. Format:\n"
    "  STATE: slotA=item1,item2 | slotB= | nowhere=item3"
)


class TrackingAgent:
    """Domain-independent full-state tracker driven by a chat backend.

    backend: any object with .chat(prompt:str)->str (ete.backends.*).
    domain:  a Domain implementation.
    """

    def __init__(self, backend, domain: Domain, verify: bool = True,
                 repair_incomplete: bool = True) -> None:
        self.backend = backend
        self.domain = domain
        self.verify = verify
        self.repair_incomplete = repair_incomplete

    def _system_preamble(self) -> str:
        slots = ", ".join(self.domain.slot_names())
        return (
            f"{self.domain.describe_rules()}\n"
            f"Slots: {slots} (plus 'nowhere').\n{_SLOT_LINE_HINT}"
        )

    def _has_reducer(self) -> bool:
        try:
            self.domain.reduce(self.domain.initial_state(), {"__probe__": True})
        except NotImplementedError:
            return False
        except Exception:
            return True  # raised for a different reason => reducer exists
        return True

    def run(self, operations: list[dict], query_entity: str) -> AgentResult:
        """Run the full sequence one operation at a time, re-emitting state each step."""
        state = self.domain.initial_state()
        truth = self.domain.initial_state()
        use_truth = self.verify and self._has_reducer()
        steps: list[StepTrace] = []
        clean = True

        for i, op in enumerate(operations):
            op_text = self.domain.render_operation(op)
            prompt = self._build_step_prompt(state, op_text, i + 1)
            try:
                out = self.backend.chat(prompt)
            except Exception as exc:
                out = f"<error: {exc}>"

            parsed = self.domain.parse_state(out)
            prev_entities = state.all_entities()
            complete = parsed is not None and prev_entities <= parsed.all_entities()

            if use_truth:
                truth = self.domain.reduce(truth, op)

            # Adopt parsed state when usable; else repair from prior state so the
            # representation never silently shrinks (the anti-ghost guard).
            if parsed is not None and (complete or not self.repair_incomplete):
                state = parsed
            elif parsed is not None and self.repair_incomplete:
                state = self._repair(parsed, prev_entities)
            # else: keep prior state unchanged

            matches = (state == truth) if use_truth else None
            if not (parsed is not None and complete and (matches is not False)):
                clean = False

            steps.append(StepTrace(
                index=i + 1, operation_text=op_text, raw_model_output=out,
                parsed_state=parsed, complete=complete,
                matches_truth=matches, truth_state=truth.copy() if use_truth else None,
            ))

        pred_loc = state.location_of(query_entity) if state else None
        truth_loc = truth.location_of(query_entity) if use_truth else None
        return AgentResult(
            final_state=state, steps=steps, clean=clean, query=query_entity,
            predicted_location=pred_loc, truth_location=truth_loc,
        )

    @staticmethod
    def _repair(parsed: State, required: set[str]) -> State:
        """Re-add any entity the model dropped, into NOWHERE, so state can't shrink."""
        repaired = parsed.copy()
        present = repaired.all_entities()
        for e in required - present:
            repaired.slots.setdefault(NOWHERE, []).append(e)
        return repaired

    def _build_step_prompt(self, state: State, op_text: str, n: int) -> str:
        return (
            f"{self._system_preamble()}\n\n"
            f"Current state:\n  STATE: {self._render_state(state)}\n\n"
            f"Step {n}: {op_text}\n"
            f"Apply this step and output the new complete STATE: line only."
        )

    @staticmethod
    def _render_state(state: State) -> str:
        parts = [f"{slot}={','.join(sorted(ents))}" for slot, ents in sorted(state.slots.items())]
        return " | ".join(parts)
