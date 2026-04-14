from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


class Strength(str, Enum):
    HARD = "hard"  # must / cannot
    SOFT = "soft"  # prefer / avoid


class ClaimType(str, Enum):
    FACTOR_VALUE = "factor_value"  # sets/observes an option's factor value
    PREFERENCE = "preference"  # prefers higher/lower for a factor
    CONSTRAINT = "constraint"  # hard bounds or prohibitions


class PreferenceDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    # Accept ISO-8601; if naive, assume UTC for deterministic behavior.
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class Source:
    name: str
    reliability: float = 0.8  # 0..1


@dataclass(frozen=True)
class Factor:
    name: str
    weight: float = 1.0
    direction: PreferenceDirection = PreferenceDirection.HIGHER_IS_BETTER
    # Optional normalization range. If missing, engine will infer from observed values.
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class Option:
    id: str
    label: str | None = None


@dataclass(frozen=True)
class Claim:
    type: ClaimType
    factor: str | None = None
    option_id: str | None = None
    # For factor_value: numeric value
    value: float | None = None
    # For constraint: "op" and "bound"
    op: Literal["<=", "<", ">=", ">", "==", "!="] | None = None
    bound: float | None = None
    # For preference: direction can override factor.direction
    direction: PreferenceDirection | None = None


@dataclass(frozen=True)
class InputStatement:
    id: str
    strength: Strength
    source: Source
    timestamp: datetime | None = None
    confidence: float = 1.0  # 0..1, independent of source reliability
    claim: Claim = field(default_factory=Claim)
    note: str | None = None


@dataclass(frozen=True)
class Conflict:
    kind: Literal[
        "hard_hard",
        "hard_soft",
        "value_value",
        "constraint_value",
        "constraint_constraint",
        "preference_preference",
        "duplicate_input",
    ]
    factor: str | None
    option_id: str | None
    statements: tuple[str, ...]  # statement ids
    summary: str
    severity: Literal["high", "medium", "low"]
    resolution: str


@dataclass
class FactorContribution:
    factor: str
    raw: float
    normalized: float
    weighted: float
    evidence_weight: float
    notes: list[str] = field(default_factory=list)


@dataclass
class OptionScore:
    option_id: str
    total: float
    by_factor: dict[str, FactorContribution]
    disqualified: bool = False
    disqualify_reasons: list[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    decision: str | None  # option_id, or None if tie/cannot decide
    status: Literal["ok", "tie", "no_valid_options", "insufficient_info"]
    scores: list[OptionScore]
    conflicts: list[Conflict]
    assumptions: list[str]
    explanation: dict[str, Any]

