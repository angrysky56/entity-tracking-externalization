#!/usr/bin/env python3
"""CLI for the entity-tracking externalization probe.

Examples
--------
    # local Ollama
    uv run python run.py --backend ollama --model gemma4:latest --n-tasks 30

    # MiniMax (Anthropic-compatible cloud endpoint); needs MINIMAX_API_KEY
    uv run python run.py --backend minimax --model MiniMax-M2.7 \
        --n-tasks 14 --n-ops 12 --n-objects 5 --remove-prob 0.45 --max-tokens 4096
"""

from __future__ import annotations

import argparse
import sys

from ete.backends import make_backend
from ete.experiment import print_report, run, save, summarize
from ete.tasks import generate_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["ollama", "minimax"], default="ollama")
    p.add_argument("--model", required=True,
                   help="model id (e.g. gemma4:latest or MiniMax-M2.7)")
    p.add_argument("--host", default=None, help="override backend base URL")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="generation cap (Ollama num_predict / MiniMax max_tokens)")
    p.add_argument("--n-tasks", type=int, default=60)
    p.add_argument("--n-ops", type=int, default=8, help="operations per task")
    p.add_argument("--n-objects", type=int, default=4)
    p.add_argument("--n-containers", type=int, default=3)
    p.add_argument("--remove-prob", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-balance", action="store_true",
                   help="do not force ~50%% removed-query tasks")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    backend = make_backend(
        args.backend, args.model, args.host, args.temperature, args.max_tokens
    )

    # Preflight checks per backend.
    if args.backend == "minimax":
        if not backend.available_models():
            print("[!] MINIMAX_API_KEY is not set in this environment.", file=sys.stderr)
            print("    export MINIMAX_API_KEY=... and retry.", file=sys.stderr)
            return 2
    else:
        available = backend.available_models()
        if not available:
            print(f"[!] Could not reach Ollama at {backend.host}. Is `ollama serve` running?",
                  file=sys.stderr)
            return 2
        if args.model not in available:
            print(f"[!] '{args.model}' not in Ollama. Available: {', '.join(available)}",
                  file=sys.stderr)
            print("    Pull it with:  ollama pull " + args.model, file=sys.stderr)
            return 2

    tasks = generate_dataset(
        n_tasks=args.n_tasks,
        base_seed=args.seed,
        balance_removed_query=not args.no_balance,
        n_ops=args.n_ops,
        n_objects=args.n_objects,
        n_containers=args.n_containers,
        remove_prob=args.remove_prob,
    )
    print(f"Generated {len(tasks)} tasks. Running 2 conditions on "
          f"{args.model} ({args.backend}) ...")

    records = run(backend, tasks)
    summary = summarize(records)
    print_report(summary)
    path = save(records, summary, args.out_dir, args.model)
    print(f"\nSaved raw results + summary -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
