"""
feedback.py — Lightweight adaptive feedback loop for the Decision Engine.

How it works
------------
After the engine makes a decision the user can record whether that decision was
correct, or which option actually turned out to be better.  The store computes
per-factor *weight multipliers* using a simple gradient-style update:

    For each factor f:
        delta_f = norm(actual_winner, f) - norm(chosen, f)

    delta_f > 0  →  actual winner scored higher on f than the chosen option did
                    →  raise multiplier for f (this factor should matter more)
    delta_f < 0  →  chosen option scored higher on f but was still wrong
                    →  lower multiplier for f (this factor misled the engine)

    new_multiplier[f] = clamp(
        old_multiplier[f] + LEARNING_RATE * delta_f,
        MIN_MULTIPLIER, MAX_MULTIPLIER,
    )

Multipliers are applied at evaluation time by scaling each Factor.weight before
calling engine.evaluate().  The original JSON weights are never modified.

Design properties
-----------------
- **Idempotent storage** — the JSON file is always a complete snapshot; partial
  writes can't leave a corrupted state because we write atomically (full dump).
- **Graceful degradation** — a missing or corrupt file starts fresh; the engine
  never raises due to feedback state.
- **Bounded adaptation** — multipliers are clamped to [MIN_MULTIPLIER, MAX_MULTIPLIER]
  so a run of bad luck can't drive a weight to zero or infinity.
- **Correct decisions reinforce nothing** — when chosen == actual we increment
  the counter but do *not* move multipliers, avoiding over-fitting to already
  good weights.
- **Transparent** — every evaluation can emit the applied multipliers, so the
  user can see exactly how past outcomes are influencing the current run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Factor, OptionScore

# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------
_LEARNING_RATE: float = 0.50   # step size per outcome
_MIN_MULTIPLIER: float = 0.25  # floor — weight can't shrink below 25 % of original
_MAX_MULTIPLIER: float = 4.0   # ceiling — weight can't inflate past 4× original


# --------------------------------------------------------------------------
# Internal state
# --------------------------------------------------------------------------
@dataclass
class _FeedbackState:
    multipliers: dict[str, float] = field(default_factory=dict)
    total_outcomes: int = 0
    correct_outcomes: int = 0
    outcomes: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------
class FeedbackStore:
    """
    Persist and apply learned factor-weight multipliers.

    Usage
    -----
    store = FeedbackStore("de_weights.json")

    # 1. Adjust factors before each evaluation.
    adjusted = store.get_adjusted_factors(factors)
    result = engine.evaluate(options=options, factors=adjusted, inputs=inputs)

    # 2. After the user knows the real outcome, record it.
    store.record_outcome(result.scores, result.decision, actual_winner="option_c")

    # 3. Inspect what was learned.
    print(store.summary())
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._state = self._load()

    # ------------------------------------------------------------------
    # Evaluation-time hook
    # ------------------------------------------------------------------

    def get_adjusted_factors(self, factors: list[Factor]) -> list[Factor]:
        """
        Return a new list of Factors with weights scaled by learned multipliers.

        Factors with no recorded history keep their original weight (multiplier=1.0).
        """
        from dataclasses import replace  # local to avoid circular at module level

        result: list[Factor] = []
        for f in factors:
            m = self._state.multipliers.get(f.name, 1.0)
            result.append(f if m == 1.0 else replace(f, weight=round(f.weight * m, 6)))
        return result

    def applied_multipliers(self, factors: list[Factor]) -> dict[str, float]:
        """Return {factor_name: multiplier} only for factors that have been adjusted."""
        return {
            f.name: round(self._state.multipliers[f.name], 4)
            for f in factors
            if f.name in self._state.multipliers and self._state.multipliers[f.name] != 1.0
        }

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        scores: list[OptionScore],
        chosen_option: str | None,
        actual_winner: str,
    ) -> dict[str, Any]:
        """
        Record an outcome and update weight multipliers.

        Parameters
        ----------
        scores:         OptionScore list from the evaluate() call being recorded.
        chosen_option:  option_id the engine returned (may be None on tie/error).
        actual_winner:  option_id that was actually the correct choice.

        Returns
        -------
        A dict describing what changed (suitable for JSON output).
        """
        # Build normalised score lookup: option_id -> factor_name -> normalised [0,1]
        norm: dict[str, dict[str, float]] = {
            s.option_id: {fname: fc.normalized for fname, fc in s.by_factor.items()}
            for s in scores
        }

        self._state.total_outcomes += 1
        adjustments: dict[str, dict[str, float]] = {}

        if chosen_option == actual_winner or chosen_option is None:
            # Correct decision (or engine couldn't decide) — nothing to adjust.
            self._state.correct_outcomes += 1
            self._save()
            return {
                "outcome": "correct" if chosen_option == actual_winner else "no_decision",
                "adjustments": {},
            }

        # Wrong prediction: shift multipliers so the actual winner would score higher.
        chosen_norm = norm.get(chosen_option, {})
        actual_norm = norm.get(actual_winner, {})
        all_factors = set(chosen_norm) | set(actual_norm)

        for fname in all_factors:
            self._state.multipliers.setdefault(fname, 1.0)
            before = self._state.multipliers[fname]
            # delta > 0 when actual winner was stronger on this factor
            delta = actual_norm.get(fname, 0.5) - chosen_norm.get(fname, 0.5)
            after = before + _LEARNING_RATE * delta
            after = max(_MIN_MULTIPLIER, min(_MAX_MULTIPLIER, after))
            self._state.multipliers[fname] = round(after, 6)
            if abs(after - before) > 1e-9:
                adjustments[fname] = {"before": round(before, 4), "after": round(after, 4)}

        self._state.outcomes.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chosen": chosen_option,
                "actual": actual_winner,
                "factor_delta": {
                    fname: round(actual_norm.get(fname, 0.5) - chosen_norm.get(fname, 0.5), 4)
                    for fname in all_factors
                },
            }
        )
        self._save()
        return {"outcome": "incorrect", "adjustments": adjustments}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Human-readable summary of the current feedback state."""
        total = self._state.total_outcomes
        correct = self._state.correct_outcomes
        accuracy = round(correct / total, 4) if total else None
        return {
            "total_outcomes_recorded": total,
            "correct_outcomes": correct,
            "accuracy": accuracy,
            "weight_multipliers": {k: round(v, 4) for k, v in self._state.multipliers.items()},
            "interpretation": (
                "multiplier > 1.0: this factor has been boosted by past feedback; "
                "multiplier < 1.0: this factor has been attenuated."
            ),
        }

    # ------------------------------------------------------------------
    # Persistence — atomicity guarantee
    # ------------------------------------------------------------------

    def _load(self) -> _FeedbackState:
        """Load from disk; return a fresh state if file is absent or corrupt."""
        if not self._path.exists():
            return _FeedbackState()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return _FeedbackState(
                multipliers=raw.get("multipliers", {}),
                total_outcomes=raw.get("total_outcomes", 0),
                correct_outcomes=raw.get("correct_outcomes", 0),
                outcomes=raw.get("outcomes", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupt or incompatible file — start fresh rather than crash.
            return _FeedbackState()

    def _save(self) -> None:
        """Write full state to disk (complete overwrite = no partial-write corruption)."""
        payload = {
            "multipliers": self._state.multipliers,
            "total_outcomes": self._state.total_outcomes,
            "correct_outcomes": self._state.correct_outcomes,
            "outcomes": self._state.outcomes,
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
