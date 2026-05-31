"""Engine tests with a scripted fake backend (no model required).

Run: uv run --extra dev pytest tests/test_agent.py
"""

from __future__ import annotations

from ete.agent import TrackingAgent
from ete.agent.core import TrackingAgent as TA
from ete.agent.domain import NOWHERE, State
from ete.agent.domains import ContainerDomain, VariableDomain


class ScriptedBackend:
    """Returns a queued response per call, ignoring the prompt."""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[str] = []

    def chat(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._outputs.pop(0) if self._outputs else "STATE: nowhere="


def test_container_clean_trace_matches_truth():
    dom = ContainerDomain(["red box", "blue box"])
    ops = [{"kind": "PUT", "obj": "apple", "container": "red box"},
           {"kind": "REMOVE", "obj": "apple"}]
    backend = ScriptedBackend([
        "STATE: red box=apple | blue box= | nowhere=",
        "STATE: red box= | blue box= | nowhere=apple",
    ])
    res = TrackingAgent(backend, dom, verify=True).run(ops, "apple")
    assert res.clean and res.correct
    assert res.predicted_location == NOWHERE == res.truth_location


def test_ghost_is_caught_as_not_clean():
    # Model fails to move apple to nowhere on REMOVE -> disagrees with reducer.
    dom = ContainerDomain(["red box", "blue box"])
    ops = [{"kind": "PUT", "obj": "apple", "container": "red box"},
           {"kind": "REMOVE", "obj": "apple"}]
    backend = ScriptedBackend([
        "STATE: red box=apple | blue box= | nowhere=",
        "STATE: red box=apple | blue box= | nowhere=",   # ghost: still in red box
    ])
    res = TrackingAgent(backend, dom, verify=True).run(ops, "apple")
    assert not res.clean                      # divergence from truth flagged
    assert res.predicted_location == "red box"
    assert res.truth_location == NOWHERE
    assert res.correct is False


def test_dropped_entity_is_repaired_to_nowhere():
    # Model drops 'apple' entirely on step 2; repair must re-add it to nowhere.
    dom = ContainerDomain(["red box"])
    ops = [{"kind": "PUT", "obj": "apple", "container": "red box"},
           {"kind": "PUT", "obj": "key", "container": "red box"}]
    backend = ScriptedBackend([
        "STATE: red box=apple | nowhere=",
        "STATE: red box=key | nowhere=",     # apple dropped from representation
    ])
    res = TrackingAgent(backend, dom, verify=False, repair_incomplete=True).run(ops, "apple")
    # apple was not erased; repair pushed it to nowhere rather than vanishing
    assert "apple" in res.final_state.all_entities()


def test_variable_domain_same_engine():
    dom = VariableDomain(["x", "y"])
    ops = [{"kind": "SET", "var": "x", "value": "5"},
           {"kind": "SET", "var": "y", "value": "9"},
           {"kind": "CLEAR", "var": "x"}]
    backend = ScriptedBackend([
        "STATE: x=5 | y= | nowhere=",
        "STATE: x=5 | y=9 | nowhere=",
        "STATE: x= | y=9 | nowhere=5",
    ])
    res = TrackingAgent(backend, dom, verify=True).run(ops, "5")
    assert res.clean
    assert res.predicted_location == NOWHERE   # value 5 was displaced when x cleared
    assert res.final_state.location_of("9") == "y"


def test_reducer_ground_truth_independent_of_model():
    # Even with a no-answer model, truth tracking via reducer stays correct.
    # Query is entity-centric: where is the value "7"? (engine tracks entities)
    dom = VariableDomain(["x"])
    ops = [{"kind": "SET", "var": "x", "value": "7"}]
    backend = ScriptedBackend(["garbage no state line"])
    res = TrackingAgent(backend, dom, verify=True).run(ops, "7")
    assert res.truth_location == "x"      # reducer knows value 7 lives in x
    assert not res.clean                  # model failed to emit usable state
