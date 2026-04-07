from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import (
    Claim,
    ClaimType,
    Factor,
    InputStatement,
    Option,
    PreferenceDirection,
    Source,
    Strength,
    parse_dt,
)


@dataclass(frozen=True)
class EvaluationRequest:
    options: list[Option]
    factors: list[Factor]
    inputs: list[InputStatement]


def load_request(path: str | Path) -> EvaluationRequest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_request(data)


def parse_request(data: dict[str, Any]) -> EvaluationRequest:
    options = [Option(**o) for o in data.get("options", [])]

    factors = []
    for f in data.get("factors", []):
        direction = f.get("direction", PreferenceDirection.HIGHER_IS_BETTER.value)
        factors.append(
            Factor(
                name=f["name"],
                weight=float(f.get("weight", 1.0)),
                direction=PreferenceDirection(direction),
                min_value=f.get("min_value", None),
                max_value=f.get("max_value", None),
            )
        )

    inputs = []
    for s in data.get("inputs", []):
        src = s.get("source", {})
        source = Source(
            name=src.get("name", "unknown"),
            reliability=float(src.get("reliability", 0.8)),
        )
        claim_raw = s.get("claim", {})
        claim_type = ClaimType(claim_raw.get("type"))
        claim = Claim(
            type=claim_type,
            factor=claim_raw.get("factor"),
            option_id=claim_raw.get("option_id"),
            value=claim_raw.get("value"),
            op=claim_raw.get("op"),
            bound=claim_raw.get("bound"),
            direction=PreferenceDirection(claim_raw["direction"])
            if claim_raw.get("direction") is not None
            else None,
        )
        inputs.append(
            InputStatement(
                id=s["id"],
                strength=Strength(s.get("strength", Strength.SOFT.value)),
                source=source,
                timestamp=parse_dt(s.get("timestamp")),
                confidence=float(s.get("confidence", 1.0)),
                claim=claim,
                note=s.get("note"),
            )
        )

    return EvaluationRequest(options=options, factors=factors, inputs=inputs)

