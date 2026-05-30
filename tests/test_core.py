"""Tests for the deterministic core (no model required).

Run with:  uv run pytest    (or)    uv run python -m pytest
"""

from __future__ import annotations

import random

from ete.grading import extract_answer, grade, normalize
from ete.tasks import NOWHERE, Op, Task, generate_dataset, generate_task


def replay(task: Task) -> dict[str, str]:
    """Independently recompute final state to confirm task.answer is correct."""
    loc = {o: NOWHERE for o in task.meta["objects"]}
    for op in task.ops:
        loc[op.obj] = NOWHERE if op.kind == "REMOVE" else op.container
    return loc


def test_ground_truth_matches_replay():
    rng = random.Random(123)
    for s in range(200):
        t = generate_task(rng, seed=s)
        assert t.answer == replay(t)[t.query_obj]


def test_balanced_dataset_has_both_query_types():
    tasks = generate_dataset(n_tasks=40, base_seed=5)
    removed = sum(t.query_obj_removed for t in tasks)
    assert 10 <= removed <= 30  # roughly balanced


def test_answer_extraction_and_normalization():
    assert extract_answer("blah\nANSWER: the Red Box.") == "red box"
    assert extract_answer("ANSWER: nowhere") == NOWHERE
    assert extract_answer("ANSWER: removed") == NOWHERE
    assert extract_answer("no marker here") is None
    assert extract_answer("ANSWER: blue box\n...\nANSWER: green box") == "green box"


def test_ghost_vs_false_remove_classification():
    t = Task(ops=[Op("PUT", "apple", "red box"), Op("REMOVE", "apple", None)],
             query_obj="apple", answer=NOWHERE, n_removes=1, query_obj_removed=True,
             meta={"objects": ["apple"], "containers": ["red box"]})
    assert grade(t, "ANSWER: red box").error_type == "ghost"
    assert grade(t, "ANSWER: nowhere").correct

    t2 = Task(ops=[Op("PUT", "key", "blue box")], query_obj="key", answer="blue box",
              n_removes=0, query_obj_removed=False,
              meta={"objects": ["key"], "containers": ["blue box"]})
    assert grade(t2, "ANSWER: nowhere").error_type == "false_remove"
    assert grade(t2, "ANSWER: green box").error_type == "stale_or_wrong"
