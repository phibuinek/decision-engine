"""
api.py — FastAPI web server for the Decision Engine.

Run:
    uvicorn decision_engine.api:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .engine import DecisionEngine
from .feedback import FeedbackStore
from .io import parse_request
from .types import OptionScore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Decision Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = DecisionEngine()
_feedback_store = FeedbackStore("de_feedback.json")

# eval_id → {"scores": list[OptionScore], "decision": str | None}
_eval_cache: dict[str, dict[str, Any]] = {}

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
_EXAMPLE_FILES = {
    "laptop":    "purchase_laptop.json",
    "conflict":  "conflicting_constraints.json",
    "duplicate": "duplicate_ids.json",
}

# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class SourceIn(BaseModel):
    name: str = "unknown"
    reliability: float = 0.8

class ClaimIn(BaseModel):
    type: str
    factor: Optional[str] = None
    option_id: Optional[str] = None
    value: Optional[float] = None
    op: Optional[str] = None
    bound: Optional[float] = None
    direction: Optional[str] = None

class InputIn(BaseModel):
    id: str
    strength: str = "soft"
    source: SourceIn = SourceIn()
    timestamp: Optional[str] = None
    confidence: float = 1.0
    claim: ClaimIn
    note: Optional[str] = None

class OptionIn(BaseModel):
    id: str
    label: Optional[str] = None

class FactorIn(BaseModel):
    name: str
    weight: float = 1.0
    direction: str = "higher_is_better"
    min_value: Optional[float] = None
    max_value: Optional[float] = None

class EvaluateRequest(BaseModel):
    options: list[OptionIn]
    factors: list[FactorIn]
    inputs: list[InputIn]
    use_feedback: bool = False

class FeedbackIn(BaseModel):
    eval_id: str
    actual_winner: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/examples")
async def list_examples():
    return list(_EXAMPLE_FILES.keys())


@app.get("/api/examples/{name}")
async def get_example(name: str):
    if name not in _EXAMPLE_FILES:
        raise HTTPException(status_code=404, detail=f"Example '{name}' not found.")
    path = _EXAMPLES_DIR / _EXAMPLE_FILES[name]
    data = json.loads(path.read_text(encoding="utf-8"))
    # Strip internal _comment keys that are for humans, not the engine.
    data.pop("_comment", None)
    return data


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    # Reconstruct the raw dict that parse_request expects.
    raw: dict[str, Any] = {
        "options": [o.model_dump() for o in req.options],
        "factors": [f.model_dump() for f in req.factors],
        "inputs": [
            {
                **i.model_dump(exclude={"claim", "source"}),
                "source": i.source.model_dump(),
                "claim": {k: v for k, v in i.claim.model_dump().items() if v is not None},
            }
            for i in req.inputs
        ],
    }

    try:
        parsed = parse_request(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request structure: {exc}")

    factors = parsed.factors
    applied_multipliers: dict[str, float] = {}
    if req.use_feedback:
        factors = _feedback_store.get_adjusted_factors(factors)
        applied_multipliers = _feedback_store.applied_multipliers(parsed.factors)

    result = _engine.evaluate(options=parsed.options, factors=factors, inputs=parsed.inputs)

    eval_id = str(uuid.uuid4())
    _eval_cache[eval_id] = {"scores": result.scores, "decision": result.decision}

    out: dict[str, Any] = {
        "eval_id": eval_id,
        "decision": result.decision,
        "status": result.status,
        "conflicts": [asdict(c) for c in result.conflicts],
        "assumptions": result.assumptions,
        "explanation": result.explanation,
    }
    if applied_multipliers:
        out["applied_multipliers"] = applied_multipliers

    return out


@app.post("/api/feedback")
async def record_feedback(req: FeedbackIn):
    cached = _eval_cache.get(req.eval_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found. Re-run evaluate first.",
        )

    update = _feedback_store.record_outcome(
        scores=cached["scores"],
        chosen_option=cached["decision"],
        actual_winner=req.actual_winner,
    )
    return {
        "outcome": update["outcome"],
        "adjustments": update["adjustments"],
        "summary": _feedback_store.summary(),
    }


@app.get("/api/feedback/summary")
async def get_feedback_summary():
    return _feedback_store.summary()


@app.delete("/api/feedback")
async def reset_feedback():
    global _feedback_store
    path = Path("de_feedback.json")
    if path.exists():
        path.unlink()
    _feedback_store = FeedbackStore("de_feedback.json")
    return {"success": True}
