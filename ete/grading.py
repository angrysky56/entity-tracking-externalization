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


def grade(task: Task, response_text: str) -> Grade:
    """Grade a response and label the error type for wrong answers.

    Error types
    -----------
    none          : correct.
    no_answer     : no parseable ANSWER line.
    ghost         : truth is NOWHERE (object was removed) but model named a
                    container -> the predicted failure of the fragile REMOVE tag
                    (removed object reported as still present).
    false_remove  : truth is a real container but model said "nowhere" -> over-
                    suppression (the opposite leak).
    stale_or_wrong: named the wrong container (stale location / contamination).
    """
    pred = extract_answer(response_text)
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
