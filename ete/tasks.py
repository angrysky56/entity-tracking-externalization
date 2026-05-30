"""Task generation for entity tracking.

A task is a sequence of natural-language state-change operations over a set of
objects and containers, followed by a "where is X?" query. The ground-truth
final location is tracked deterministically so answers can be graded.

Operations
----------
PUT    : object enters a container (writes state)
MOVE   : object relocates from its current container to another
REMOVE : object leaves all containers -> truth becomes "nowhere"

The REMOVE op is the stress case: arXiv:2605.30233 finds models implement it as
a fragile global suppression tag, predicting "ghost" errors (a removed object
reported as still present) that accumulate as removals pile up.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

NOWHERE = "nowhere"

# Distinct, unambiguous nouns. Kept concrete so tokenization is clean and the
# model is not helped/hurt by semantic overlap between object and container.
OBJECTS = [
    "apple", "key", "book", "candle", "coin", "spoon", "ring", "stamp",
    "marble", "feather", "button", "thimble", "cork", "pebble", "acorn",
]
CONTAINERS = [
    "red box", "blue box", "green box", "yellow drawer", "black basket",
    "white crate", "brown bag", "gray bin",
]


@dataclass
class Op:
    """A single state-change operation with a rendered natural-language form."""

    kind: str  # "PUT" | "MOVE" | "REMOVE"
    obj: str
    container: str | None  # destination for PUT/MOVE; None for REMOVE

    def render(self) -> str:
        if self.kind == "PUT":
            return f"Put the {self.obj} in the {self.container}."
        if self.kind == "MOVE":
            return f"Move the {self.obj} to the {self.container}."
        return f"Take the {self.obj} out."  # REMOVE


@dataclass
class Task:
    """A generated entity-tracking problem with ground truth and metadata."""

    ops: list[Op]
    query_obj: str
    answer: str  # ground-truth final location, or NOWHERE
    n_removes: int  # total REMOVE ops in the trace (accumulation stressor)
    query_obj_removed: bool  # was the queried object ever removed?
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def render_ops(self) -> str:
        """Numbered operation list, one per line."""
        return "\n".join(f"{i + 1}. {op.render()}" for i, op in enumerate(self.ops))


def generate_task(
    rng: random.Random,
    n_ops: int = 8,
    n_objects: int = 4,
    n_containers: int = 3,
    remove_prob: float = 0.25,
    seed: int = 0,
) -> Task:
    """Generate one task with a deterministically tracked final state.

    Args:
        rng: seeded RNG for reproducibility.
        n_ops: number of operations in the trace.
        n_objects: distinct objects in play.
        n_containers: distinct containers in play.
        remove_prob: probability an eligible step is a REMOVE.
        seed: recorded on the task for traceability.
    """
    objects = rng.sample(OBJECTS, k=n_objects)
    containers = rng.sample(CONTAINERS, k=n_containers)

    # location[obj] = container name or NOWHERE (not yet placed / removed)
    location: dict[str, str] = {o: NOWHERE for o in objects}
    ops: list[Op] = []
    n_removes = 0

    for _ in range(n_ops):
        present = [o for o in objects if location[o] != NOWHERE]
        absent = [o for o in objects if location[o] == NOWHERE]

        # Decide op kind. REMOVE/MOVE require an object already placed.
        roll = rng.random()
        if present and roll < remove_prob:
            obj = rng.choice(present)
            ops.append(Op("REMOVE", obj, None))
            location[obj] = NOWHERE
            n_removes += 1
        elif present and roll < remove_prob + 0.35:
            obj = rng.choice(present)
            dest = rng.choice([c for c in containers if c != location[obj]] or containers)
            ops.append(Op("MOVE", obj, dest))
            location[obj] = dest
        else:
            # PUT: prefer an absent object so state grows; else re-place one.
            obj = rng.choice(absent or objects)
            dest = rng.choice(containers)
            ops.append(Op("PUT", obj, dest))
            location[obj] = dest

    query_obj = rng.choice(objects)
    return Task(
        ops=ops,
        query_obj=query_obj,
        answer=location[query_obj],
        n_removes=n_removes,
        query_obj_removed=any(
            op.kind == "REMOVE" and op.obj == query_obj for op in ops
        ),
        seed=seed,
        meta={"objects": objects, "containers": containers},
    )


def generate_dataset(
    n_tasks: int = 60,
    base_seed: int = 7,
    balance_removed_query: bool = True,
    **task_kwargs,
) -> list[Task]:
    """Generate a reproducible list of tasks.

    When ``balance_removed_query`` is set, roughly half the tasks are forced to
    query an object that was removed (answer == NOWHERE), so the REMOVE failure
    mode is well represented rather than rare.
    """
    rng = random.Random(base_seed)
    tasks: list[Task] = []
    attempts = 0
    want_removed = True

    while len(tasks) < n_tasks and attempts < n_tasks * 50:
        attempts += 1
        t = generate_task(rng, seed=base_seed + attempts, **task_kwargs)
        if balance_removed_query and t.query_obj_removed != want_removed:
            continue
        tasks.append(t)
        want_removed = not want_removed
    return tasks
