"""Prompt construction for the two experimental conditions.

Both conditions end with the same machine-readable marker so grading is
identical and the comparison is fair:

    ANSWER: <container name or "nowhere">

DIRECT
    Present the operations, ask the question, demand only the final answer.
    This pushes the model toward the parallel final-token aggregation strategy
    documented in arXiv:2605.30233 (no room to track state step by step).

EXTERNALIZED
    Force the model to rewrite the *full* state of every container after each
    operation before answering. This externalizes the sequential computation
    into tokens, which is the bridge the synthesis page proposes for letting a
    test-time sampler (and the model itself) act on per-step state.
"""

from __future__ import annotations

from .tasks import Task

_COMMON_RULES = (
    "There are containers and objects. Each object is in at most one container. "
    'Taking an object "out" removes it from all containers (it is then nowhere). '
    "Answer the question about the final state after all steps."
)

_ANSWER_SPEC = (
    'Finish with exactly one line in this form:\nANSWER: <container name, or the word "nowhere">'
)


def build_direct_prompt(task: Task) -> str:
    """Direct condition: answer with no intermediate reasoning."""
    return (
        f"{_COMMON_RULES}\n\n"
        f"Steps:\n{task.render_ops()}\n\n"
        f"Question: Where is the {task.query_obj} at the end?\n"
        f"Do not explain. {_ANSWER_SPEC}"
    )


def build_externalized_prompt(task: Task) -> str:
    """Externalized condition: rewrite full state after every step, then answer."""
    return (
        f"{_COMMON_RULES}\n\n"
        f"Steps:\n{task.render_ops()}\n\n"
        "Work step by step. After each numbered step, write the complete current "
        "contents on its own line in this exact form:\n"
        "  Step <n>: <container>=<object or empty>; <container>=<object or empty>; ... "
        "; nowhere=<objects not in any container>\n"
        "Update every container each time. Track removed objects under 'nowhere'.\n\n"
        f"After the final step, answer: Where is the {task.query_obj} at the end?\n"
        f"{_ANSWER_SPEC}"
    )


CONDITIONS: dict[str, "callable"] = {
    "direct": build_direct_prompt,
    "externalized": build_externalized_prompt,
}
