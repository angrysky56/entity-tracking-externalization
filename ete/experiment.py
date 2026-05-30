"""Experiment runner and analysis.

Runs both conditions over a shared task set and reports the contrasts that bear
on the hypothesis:

  H: externalizing state into CoT routes around the fragile REMOVE mechanism.

Decisive signals
  * accuracy gain (externalized - direct) is *larger* on removed-query items
    than on present-query items;
  * ghost-error rate drops sharply under externalization;
  * direct accuracy degrades as n_removes grows, externalized stays flatter.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import OllamaBackend
from .grading import Grade, grade
from .prompts import CONDITIONS
from .tasks import Task, generate_dataset


@dataclass
class Record:
    """One (task, condition) trial."""

    idx: int
    condition: str
    query_obj_removed: bool
    n_removes: int
    correct: bool
    error_type: str
    predicted: str | None
    truth: str
    response: str


def run(
    backend: OllamaBackend,
    tasks: list[Task],
    conditions: list[str] | None = None,
    progress: bool = True,
) -> list[Record]:
    """Run every condition over every task; return flat trial records."""
    conditions = conditions or list(CONDITIONS)
    records: list[Record] = []
    total = len(tasks) * len(conditions)
    done = 0

    for idx, task in enumerate(tasks):
        for cond in conditions:
            prompt = CONDITIONS[cond](task)
            try:
                text = backend.chat(prompt)
            except Exception as exc:  # network/model errors -> recorded, not fatal
                text = f"<error: {exc}>"
            g: Grade = grade(task, text)
            records.append(
                Record(
                    idx=idx,
                    condition=cond,
                    query_obj_removed=task.query_obj_removed,
                    n_removes=task.n_removes,
                    correct=g.correct,
                    error_type=g.error_type,
                    predicted=g.predicted,
                    truth=g.truth,
                    response=text,
                )
            )
            done += 1
            if progress:
                print(f"  [{done}/{total}] task {idx} / {cond} -> "
                      f"{'OK' if g.correct else g.error_type}", flush=True)
    return records


def _acc(recs: list[Record]) -> float:
    return statistics.mean(r.correct for r in recs) if recs else float("nan")


def summarize(records: list[Record]) -> dict:
    """Aggregate the decisive contrasts into a plain dict."""
    out: dict = {"conditions": {}}
    conds = sorted({r.condition for r in records})

    for cond in conds:
        cr = [r for r in records if r.condition == cond]
        removed = [r for r in cr if r.query_obj_removed]
        present = [r for r in cr if not r.query_obj_removed]
        out["conditions"][cond] = {
            "n": len(cr),
            "accuracy_overall": _acc(cr),
            "accuracy_removed_query": _acc(removed),
            "accuracy_present_query": _acc(present),
            "error_types": dict(Counter(r.error_type for r in cr)),
            "ghost_rate": statistics.mean(r.error_type == "ghost" for r in cr) if cr else 0.0,
        }

    # Headline contrast: externalization gain on removed vs present queries.
    if {"direct", "externalized"} <= set(conds):
        d, e = out["conditions"]["direct"], out["conditions"]["externalized"]
        out["headline"] = {
            "gain_removed_query": e["accuracy_removed_query"] - d["accuracy_removed_query"],
            "gain_present_query": e["accuracy_present_query"] - d["accuracy_present_query"],
            "ghost_rate_direct": d["ghost_rate"],
            "ghost_rate_externalized": e["ghost_rate"],
            "interpretation": (
                "Larger gain on removed-query items + ghost-rate drop => "
                "externalization routes around the fragile REMOVE tag (recombination-OOD, "
                "no retraining needed). Similar gains across subsets => the failure is "
                "preserved, pointing to capability-OOD."
            ),
        }
    return out


def save(records: list[Record], summary: dict, out_dir: str | Path, model: str) -> Path:
    """Write raw records + summary to a timestamped JSON file; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_").replace(":", "-")
    path = out_dir / f"results-{safe_model}-{stamp}.json"
    path.write_text(
        json.dumps(
            {"model": model, "summary": summary, "records": [asdict(r) for r in records]},
            indent=2,
        )
    )
    return path


def print_report(summary: dict) -> None:
    """Human-readable console report of the headline contrasts."""
    print("\n=== Accuracy by condition ===")
    for cond, s in summary["conditions"].items():
        print(
            f"{cond:>13}: overall {s['accuracy_overall']:.2f} | "
            f"removed-query {s['accuracy_removed_query']:.2f} | "
            f"present-query {s['accuracy_present_query']:.2f} | "
            f"ghost {s['ghost_rate']:.2f}"
        )
    h = summary.get("headline")
    if h:
        print("\n=== Headline contrast (externalized - direct) ===")
        print(f"  gain on removed-query items : {h['gain_removed_query']:+.2f}")
        print(f"  gain on present-query items : {h['gain_present_query']:+.2f}")
        print(f"  ghost rate  direct -> ext   : {h['ghost_rate_direct']:.2f} -> "
              f"{h['ghost_rate_externalized']:.2f}")
        print("\n" + h["interpretation"])
