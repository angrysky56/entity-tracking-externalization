#!/usr/bin/env python3
"""Difficulty staircase: is externalization a capability *extender*?

Runs both conditions across a ladder of trace lengths (n_ops) on one model and
reports the direct-vs-externalized accuracy curve. The question this answers:
as tasks get harder and `direct` degrades, does `externalized` hold up? A
widening gap means externalization extends the usable difficulty range, not just
a fixed boost. Higher n_ops also accumulates more REMOVEs, pushing into the
ghost-dominated regime where the fragile-REMOVE hypothesis is sharpest.

Trace length is the single varied axis (objects/containers fixed) for clean
attribution.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ete.backends import make_backend
from ete.experiment import run, summarize
from ete.tasks import generate_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["ollama", "minimax"], default="ollama")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default=None)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--n-tasks", type=int, default=20, help="tasks per level")
    p.add_argument("--ladder", default="8,16,24,32,40",
                   help="comma-separated n_ops difficulty levels")
    p.add_argument("--n-objects", type=int, default=6)
    p.add_argument("--n-containers", type=int, default=4)
    p.add_argument("--remove-prob", type=float, default=0.45)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    return run_sweep(args)


def run_sweep(args) -> int:
    backend = make_backend(
        args.backend, args.model, args.host, args.temperature, args.max_tokens
    )
    ladder = [int(x) for x in args.ladder.split(",")]
    levels = []

    for n_ops in ladder:
        tasks = generate_dataset(
            n_tasks=args.n_tasks,
            base_seed=args.seed,
            balance_removed_query=True,
            n_ops=n_ops,
            n_objects=args.n_objects,
            n_containers=args.n_containers,
            remove_prob=args.remove_prob,
        )
        print(f"\n### n_ops={n_ops} ({len(tasks)} tasks, {len(tasks) * 2} trials) ###",
              flush=True)
        records = run(backend, tasks, progress=False)
        s = summarize(records)
        d, e = s["conditions"]["direct"], s["conditions"]["externalized"]
        levels.append({"n_ops": n_ops, "summary": s})
        print(f"  direct={d['accuracy_overall']:.2f}  ext={e['accuracy_overall']:.2f}  "
              f"gain={e['accuracy_overall'] - d['accuracy_overall']:+.2f}  | "
              f"ghost {d['ghost_rate']:.2f}->{e['ghost_rate']:.2f}", flush=True)

    print("\n=== Difficulty staircase ===")
    print(f"{'n_ops':>6} {'direct':>7} {'ext':>7} {'gain':>7} {'d_ghost':>8} {'e_ghost':>8}")
    for lv in levels:
        d, e = lv["summary"]["conditions"]["direct"], lv["summary"]["conditions"]["externalized"]
        print(f"{lv['n_ops']:>6} {d['accuracy_overall']:>7.2f} {e['accuracy_overall']:>7.2f} "
              f"{e['accuracy_overall'] - d['accuracy_overall']:>+7.2f} "
              f"{d['ghost_rate']:>8.2f} {e['ghost_rate']:>8.2f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = args.model.replace("/", "_").replace(":", "-")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sweep-{safe}-{stamp}.json"
    path.write_text(json.dumps(
        {"model": args.model, "backend": args.backend, "args": vars(args), "levels": levels},
        indent=2,
    ))
    print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
