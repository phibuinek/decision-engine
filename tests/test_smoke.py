import json
from pathlib import Path

from decision_engine.engine import DecisionEngine
from decision_engine.io import load_request


def test_purchase_laptop_smoke():
    req = load_request(Path("examples") / "purchase_laptop.json")
    result = DecisionEngine().evaluate(options=req.options, factors=req.factors, inputs=req.inputs)
    assert result.status in ("ok", "tie", "no_valid_options", "insufficient_info")
    # In the bundled example, B violates hard budget constraint so should not win.
    assert result.decision in (None, "a", "c")


def test_conflicting_constraints_smoke():
    req = load_request(Path("examples") / "conflicting_constraints.json")
    result = DecisionEngine().evaluate(options=req.options, factors=req.factors, inputs=req.inputs)
    assert result.status == "no_valid_options"
    assert any(c.kind == "constraint_constraint" for c in result.conflicts)


def test_cli_json_serializable():
    req = load_request(Path("examples") / "purchase_laptop.json")
    result = DecisionEngine().evaluate(options=req.options, factors=req.factors, inputs=req.inputs)
    payload = {
        "decision": result.decision,
        "status": result.status,
        "conflicts": [c.__dict__ for c in result.conflicts],
        "assumptions": result.assumptions,
        "explanation": result.explanation,
    }
    json.dumps(payload)

