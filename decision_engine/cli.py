from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import DecisionEngine
from .io import load_request


def main() -> int:
    p = argparse.ArgumentParser(description="Deterministic decision engine (no AI APIs).")
    p.add_argument("request_json", help="Path to evaluation request JSON.")
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (default true if stdout is a terminal).",
    )
    args = p.parse_args()

    req = load_request(args.request_json)
    engine = DecisionEngine()
    result = engine.evaluate(options=req.options, factors=req.factors, inputs=req.inputs)

    out = {
        "decision": result.decision,
        "status": result.status,
        "conflicts": [c.__dict__ for c in result.conflicts],
        "assumptions": result.assumptions,
        "explanation": result.explanation,
    }
    pretty = args.pretty
    text = json.dumps(out, indent=2 if pretty else None, sort_keys=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

