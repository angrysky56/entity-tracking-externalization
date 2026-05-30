#!/usr/bin/env python3
"""Competence-frontier finder: the difficulty where each approach breaks.

Instead of "accuracy at an arbitrary difficulty D", this measures the quantity
that actually matters: the *frontier* — the largest trace length (n_ops) at
which a condition still solves the task above a pass threshold. The headline is
the **frontier shift** between conditions: how many more steps a model can
handle *because* it externalized. This auto-calibrates to each model's ceiling
(no budget wasted on trivial lengths) and makes models directly comparable.

Method (per condition): walk a difficulty ladder upward; at each rung run
`n_tasks` trials; stop when accuracy drops below `--pass-threshold`. The frontier
is the last rung at/above threshold. This is the standard length-generalization
curve, reduced to its crossing point.

Conditions come from ete.prompts.CONDITIONS: direct, externalized (full state
dump), delta (minimal-edit, Turing-Program style). Comparing externalized vs
delta separates "use external memory" from "re-state the whole world each step".
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ete.backends import make_backend
from ete.experiment import run, summarize
from ete.prompts import CONDITIONS
from ete.tasks import generate_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["ollama", "minimax"], default="ollama")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default=None)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--conditions", default="direct,delta,externalized",
                   help="comma-separated subset of: direct, externalized, delta")
    p.add_argument("--ladder", default="6,10,16,24,36,52",
                   help="n_ops rungs, ascending")
    p.add_argument("--n-tasks", type=int, default=12, help="trials per rung")
    p.add_argument("--pass-threshold", type=float, default=0.6)
    p.add_argument("--n-objects", type=int, default=6)
    p.add_argument("--n-containers", type=int, default=4)
    p.add_argument("--remove-prob", type=float, default=0.45)
    p.add_argument("--similar-names", action="store_true",
                   help="use confusable object names (contamination stressor)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--think", choices=["on", "off", "default"], default="default")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    return run_frontier(args)


def _make_backend(args, think_flag: bool | None):
    if args.backend == "ollama":
        from ete.backends import OllamaBackend
        return OllamaBackend(
            model=args.model, host=args.host or "http://localhost:11434",
            temperature=args.temperature, num_predict=args.max_tokens,
            num_ctx=16384, think=think_flag, timeout=600.0,
        )
    return make_backend(args.backend, args.model, args.host,
                        args.temperature, args.max_tokens)


def run_frontier(args) -> int:
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip() in CONDITIONS]
    ladder = [int(x) for x in args.ladder.split(",")]
    think_flag = {"on": True, "off": False, "default": None}[args.think]
    backend = _make_backend(args, think_flag)

    # frontier[cond] = highest n_ops at/above pass threshold (0 if it fails rung 1)
    frontier: dict[str, int] = {c: 0 for c in conditions}
    curve: dict[str, list] = {c: [] for c in conditions}
    active = set(conditions)

    for n_ops in ladder:
        if not active:
            break
        tasks = generate_dataset(
            n_tasks=args.n_tasks, base_seed=args.seed, balance_removed_query=True,
            n_ops=n_ops, n_objects=args.n_objects, n_containers=args.n_containers,
            remove_prob=args.remove_prob, similar_names=args.similar_names,
        )
        print(f"\n### n_ops={n_ops}  (active: {sorted(active)}) ###", flush=True)
        for cond in list(active):
            recs = run(backend, tasks, conditions=[cond],
                       progress=False, verbose=args.verbose)
            s = summarize(recs)["conditions"][cond]
            acc = s["accuracy_overall"]
            curve[cond].append({"n_ops": n_ops, "acc": acc,
                                "ghost": s["ghost_rate"], "errs": s["error_types"]})
            print(f"  {cond:>13}: acc={acc:.2f} ghost={s['ghost_rate']:.2f} "
                  f"errs={s['error_types']}", flush=True)
            if acc >= args.pass_threshold:
                frontier[cond] = n_ops
            else:
                active.discard(cond)  # broke; stop testing harder rungs
    report(frontier, curve, args)
    return 0


def report(frontier: dict[str, int], curve: dict[str, list], args) -> None:
    print("\n=== Competence frontier (highest n_ops at/above "
          f"{args.pass_threshold:.0%}) ===")
    for cond, fr in frontier.items():
        print(f"  {cond:>13}: {fr}")

    base = frontier.get("direct")
    if base is not None:
        print("\n=== Frontier shift vs direct (extra steps unlocked) ===")
        for cond, fr in frontier.items():
            if cond != "direct":
                print(f"  {cond:>13}: {fr - base:+d}")
        print("\nInterpretation: a positive shift = externalization lets the model "
              "handle longer traces than it can natively (capability extension). "
              "delta >= externalized would mean the minimal-edit format beats the "
              "verbose full-dump (format, not memory, was the bottleneck).")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = args.model.replace("/", "_").replace(":", "-")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"frontier-{safe}-{stamp}.json"
    path.write_text(json.dumps(
        {"model": args.model, "backend": args.backend, "args": vars(args),
         "frontier": frontier, "curve": curve},
        indent=2,
    ))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
