#!/usr/bin/env python3
"""2x2: reasoning {on, off} x prompt {direct, externalized}.

The real test of "does explicit externalization substitute for free-form
reasoning, and could we switch reasoning off when it triggers?" Run on a
toggleable reasoning model (e.g. deepseek-r1 via Ollama, which honors
`think: false`).

Four cells on one shared task set:

    reason_on  + direct        -> native reasoning baseline (what R1 does now)
    reason_on  + externalized  -> reasoning + explicit scaffold (redundant? additive?)
    reason_off + direct        -> the non-reasoning floor
    reason_off + externalized  -> the proposal: scaffold replaces reasoning

Decisive reads:
  * (reason_off + externalized) >= (reason_on + direct)  => the scaffold can
    REPLACE reasoning on this task class -> switch reasoning off, save the
    reasoning tokens, keep accuracy.
  * (reason_on + externalized) >> (reason_on + direct)   => scaffold ADDS to
    reasoning -> keep both.
  * reason_off cells collapse while reason_on holds       => reasoning is doing
    the real work here; scaffold alone insufficient.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ete.backends import OllamaBackend
from ete.experiment import run, summarize
from ete.tasks import generate_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="toggleable reasoning model, e.g. deepseek-r1:latest")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--n-ops", type=int, default=16)
    p.add_argument("--n-objects", type=int, default=5)
    p.add_argument("--n-containers", type=int, default=4)
    p.add_argument("--remove-prob", type=float, default=0.45)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--verbose", action="store_true",
                   help="echo each full model response (reasoning trace / state rewrite)")
    args = p.parse_args()
    return run_2x2(args)


def run_2x2(args) -> int:
    tasks = generate_dataset(
        n_tasks=args.n_tasks,
        base_seed=args.seed,
        balance_removed_query=True,
        n_ops=args.n_ops,
        n_objects=args.n_objects,
        n_containers=args.n_containers,
        remove_prob=args.remove_prob,
    )
    print(f"Generated {len(tasks)} tasks (n_ops={args.n_ops}). "
          f"4 cells x {len(tasks)*2} trials each on {args.model}.")

    cells: dict[str, dict] = {}
    all_records: dict[str, list] = {}
    for think in (True, False):
        # max_tokens must be generous when reasoning is on (trace + answer).
        cap = args.max_tokens if think else max(1024, args.max_tokens // 2)
        backend = OllamaBackend(
            model=args.model, host=args.host, temperature=args.temperature,
            num_predict=cap, num_ctx=16384, think=think, timeout=600.0,
        )
        tag = "reason_on" if think else "reason_off"
        print(f"\n### {tag} (think={think}, cap={cap}) ###", flush=True)
        records = run(backend, tasks, progress=True, verbose=args.verbose)
        s = summarize(records)
        # Keep raw records per cell so wrong answers (esp. ghosts) are auditable.
        all_records[tag] = [r.__dict__ for r in records]
        # summarize() keys by prompt condition: 'direct' / 'externalized'
        for cond in ("direct", "externalized"):
            v = s["conditions"][cond]
            key = f"{tag}+{cond}"
            cells[key] = v
            print(f"  {key:>28}: acc={v['accuracy_overall']:.2f} "
                  f"removed={v['accuracy_removed_query']:.2f} ghost={v['ghost_rate']:.2f} "
                  f"errs={v['error_types']}", flush=True)

    print_grid(cells)
    save_2x2(cells, args, all_records)
    return 0


def print_grid(cells: dict[str, dict]) -> None:
    def acc(k: str) -> float:
        return cells[k]["accuracy_overall"]

    print("\n=== 2x2 accuracy grid ===")
    print(f"{'':>12} {'direct':>10} {'externalized':>14}")
    print(f"{'reason_on':>12} {acc('reason_on+direct'):>10.2f} {acc('reason_on+externalized'):>14.2f}")
    print(f"{'reason_off':>12} {acc('reason_off+direct'):>10.2f} {acc('reason_off+externalized'):>14.2f}")

    native = acc("reason_on+direct")
    proposal = acc("reason_off+externalized")
    both = acc("reason_on+externalized")
    print("\n=== reads ===")
    print(f"  native (reason_on+direct)          : {native:.2f}")
    print(f"  proposal (reason_off+externalized) : {proposal:.2f}  "
          f"=> {'CAN replace reasoning' if proposal >= native - 1e-9 else 'cannot fully replace'}")
    print(f"  both (reason_on+externalized)      : {both:.2f}  "
          f"=> scaffold {'+adds over' if both > native + 1e-9 else 'no gain over'} reasoning-direct")
    print(f"  reasoning-only lift (on-off, direct): "
          f"{native - acc('reason_off+direct'):+.2f}")
    print(f"  scaffold-only lift (ext-direct, off): "
          f"{proposal - acc('reason_off+direct'):+.2f}")


def save_2x2(cells: dict[str, dict], args, all_records: dict[str, list] | None = None) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = args.model.replace("/", "_").replace(":", "-")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"twobytwo-{safe}-{stamp}.json"
    path.write_text(json.dumps(
        {"model": args.model, "args": vars(args), "cells": cells,
         "records": all_records or {}},
        indent=2,
    ))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
