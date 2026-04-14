from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import DecisionEngine
from .feedback import FeedbackStore
from .io import load_request


def main() -> int:
    p = argparse.ArgumentParser(
        description="Deterministic decision engine (no AI APIs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Basic evaluation
  decision-engine request.json --pretty

  # Evaluation with feedback-adjusted weights
  decision-engine request.json --feedback-store ./de_weights.json --pretty

  # Record that the engine was wrong and option_c was actually best
  decision-engine request.json --feedback-store ./de_weights.json --actual-winner option_c

  # Inspect what the feedback store has learned so far
  decision-engine --show-feedback --feedback-store ./de_weights.json
""",
    )
    p.add_argument(
        "request_json",
        nargs="?",
        help="Path to evaluation request JSON (required unless --show-feedback).",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    p.add_argument(
        "--feedback-store",
        metavar="PATH",
        help=(
            "Path to the feedback store JSON. "
            "When supplied, factor weights are adjusted by learned multipliers "
            "before evaluation."
        ),
    )
    p.add_argument(
        "--actual-winner",
        metavar="OPTION_ID",
        help=(
            "Record that OPTION_ID was the correct choice for this request, "
            "updating the feedback store's weight multipliers. "
            "Requires --feedback-store."
        ),
    )
    p.add_argument(
        "--show-feedback",
        action="store_true",
        help="Print the feedback store summary and exit (requires --feedback-store).",
    )
    args = p.parse_args()

    indent = 2 if args.pretty else None

    # ------------------------------------------------------------------
    # --show-feedback: inspect learned state, no evaluation needed.
    # ------------------------------------------------------------------
    if args.show_feedback:
        if not args.feedback_store:
            print("Error: --show-feedback requires --feedback-store.", file=sys.stderr)
            return 1
        store = FeedbackStore(args.feedback_store)
        print(json.dumps(store.summary(), indent=2))
        return 0

    # ------------------------------------------------------------------
    # Normal evaluation path.
    # ------------------------------------------------------------------
    if not args.request_json:
        p.print_help()
        return 1

    req = load_request(args.request_json)

    # Apply feedback-adjusted weights if a store is provided.
    store: FeedbackStore | None = None
    factors = req.factors
    applied_multipliers: dict = {}
    if args.feedback_store:
        store = FeedbackStore(args.feedback_store)
        factors = store.get_adjusted_factors(req.factors)
        applied_multipliers = store.applied_multipliers(req.factors)

    engine = DecisionEngine()
    result = engine.evaluate(options=req.options, factors=factors, inputs=req.inputs)

    out: dict = {
        "decision": result.decision,
        "status": result.status,
        "conflicts": [c.__dict__ for c in result.conflicts],
        "assumptions": result.assumptions,
        "explanation": result.explanation,
    }

    # ------------------------------------------------------------------
    # --actual-winner: record outcome, update multipliers.
    # ------------------------------------------------------------------
    feedback_info: dict = {}
    if args.actual_winner:
        if not store:
            print("Error: --actual-winner requires --feedback-store.", file=sys.stderr)
            return 1
        update = store.record_outcome(
            scores=result.scores,
            chosen_option=result.decision,
            actual_winner=args.actual_winner,
        )
        feedback_info = {
            "store": str(args.feedback_store),
            "applied_multipliers": applied_multipliers,
            "actual_winner_recorded": args.actual_winner,
            "outcome": update.get("outcome"),
            "weight_adjustments": update.get("adjustments", {}),
        }
    elif applied_multipliers:
        feedback_info = {
            "store": str(args.feedback_store),
            "applied_multipliers": applied_multipliers,
        }

    if feedback_info:
        out["feedback"] = feedback_info

    print(json.dumps(out, indent=indent, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
