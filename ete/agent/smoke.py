"""Tiny live smoke test of the generalized agent against a real Ollama model.

Not a pytest (needs a running model). Run directly:
    uv run python -m ete.agent.smoke --model granite4.1:3b
"""

from __future__ import annotations

import argparse

from ete.backends import OllamaBackend
from ete.agent import TrackingAgent
from ete.agent.domains import ContainerDomain, VariableDomain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="granite4.1:3b")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()
    backend = OllamaBackend(model=args.model, host=args.host, temperature=0.0,
                            num_predict=512, timeout=180.0)

    print("=== ContainerDomain ===")
    dom = ContainerDomain(["red box", "blue box", "green box"])
    ops = [{"kind": "PUT", "obj": "apple", "container": "red box"},
           {"kind": "PUT", "obj": "key", "container": "blue box"},
           {"kind": "MOVE", "obj": "apple", "container": "green box"},
           {"kind": "REMOVE", "obj": "apple"}]
    res = TrackingAgent(backend, dom, verify=True).run(ops, "apple")
    print(f"  predicted={res.predicted_location} truth={res.truth_location} "
          f"clean={res.clean} correct={res.correct}")

    print("=== VariableDomain (same engine) ===")
    dom2 = VariableDomain(["x", "y", "z"])
    ops2 = [{"kind": "SET", "var": "x", "value": "alpha"},
            {"kind": "SET", "var": "y", "value": "beta"},
            {"kind": "CLEAR", "var": "x"}]
    res2 = TrackingAgent(backend, dom2, verify=True).run(ops2, "alpha")
    print(f"  predicted={res2.predicted_location} truth={res2.truth_location} "
          f"clean={res2.clean} correct={res2.correct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
