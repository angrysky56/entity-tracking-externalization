"""Grading and failure-mode classification.

Answer extraction is uniform across conditions: take the last ``ANSWER:`` line.
Correctness is exact-match after normalization. For *wrong* answers we classify
the error so the REMOVE-specific "ghost" signature can be measured separately
from generic mistakes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tasks import NOWHERE, Task

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def normalize(s: str) -> str:
    """Lowercase, strip punctuation/articles/filler so 'the Red Box.' == 'red box'."""
    s = s.strip().lower().strip(".'\"` ")
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    if s in {"none", "no container", "not in any container", "removed", "out"}:
        return NOWHERE
    return s


def extract_answer(text: str) -> str | None:
    """Return the normalized content of the final ANSWER: line, or None."""
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return normalize(matches[-1])


@dataclass
class Grade:
    """Outcome of grading one model response against a task."""

    correct: bool
    predicted: str | None
    truth: str
    error_type: str  # "none" | "no_answer" | "ghost" | "stale_or_wrong" | "false_remove"


def salvage_from_trace(task: Task, text: str) -> str | None:
    """Recover the query object's location from a *complete* step trace that
    omitted the ANSWER line.

    Only salvages when the trace reached the final step (a 'Step <n_ops>' line is
    present); an incomplete/truncated trace stays None so it is honestly counted
    as no_answer rather than graded on stale state. Handles both formats:
      full-dump:  'Step 12: red box=apple; ...; nowhere=key, coin'
      delta:      'Step 12: apple -> red box'   /   'apple -> nowhere'
    """
    n = len(task.ops)
    obj = task.query_obj.lower()
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    step_re = re.compile(rf"step\s*{n}\b(.*)", re.IGNORECASE)
    final = next((m.group(1) for l in reversed(lines) if (m := step_re.match(l))), None)
    if final is None:
        return None
    seg = final.lower()
    # delta format: "<obj> -> <loc>"
    m = re.search(rf"{re.escape(obj)}\s*->\s*([^;,\n]+)", seg)
    if m:
        return normalize(m.group(1))
    # full-dump: object appears under a "container=...obj..." or "nowhere=...obj..."
    for chunk in seg.split(";"):
        if "=" in chunk and obj in chunk:
            loc = chunk.split("=", 1)[0].strip()
            return NOWHERE if loc == "nowhere" else normalize(loc)
    return None


def grade(task: Task, response_text: str) -> Grade:
    """Grade a response and label the error type for wrong answers.

    Error types
    -----------
    none          : correct.
    no_answer     : no parseable ANSWER line and no salvageable complete trace.
    ghost         : truth is NOWHERE (object was removed) but model named a
                    container -> the predicted failure of the fragile REMOVE tag
                    (removed object reported as still present).
    false_remove  : truth is a real container but model said "nowhere" -> over-
                    suppression (the opposite leak).
    stale_or_wrong: named the wrong container (stale location / contamination).
    """
    pred = extract_answer(response_text)
    if pred is None:
        # No ANSWER line: try to recover from a completed step trace before
        # giving up, so a verbose-but-finished response is not lost as no_answer.
        pred = salvage_from_trace(task, response_text)
    truth = normalize(task.answer)

    if pred is None:
        return Grade(False, None, truth, "no_answer")
    if pred == truth:
        return Grade(True, pred, truth, "none")
    if truth == NOWHERE and pred != NOWHERE:
        return Grade(False, pred, truth, "ghost")
    if truth != NOWHERE and pred == NOWHERE:
        return Grade(False, pred, truth, "false_remove")
    return Grade(False, pred, truth, "stale_or_wrong")
