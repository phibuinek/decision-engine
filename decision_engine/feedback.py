"""
feedback.py — Adaptive feedback loop for the Decision Engine.

Algorithm
---------
After each wrong decision the store adjusts per-factor weight multipliers:

    delta_f = norm(actual_winner, f) - norm(chosen, f)
    new_multiplier[f] = clamp(
        old_multiplier[f] + LEARNING_RATE * delta_f,
        MIN_MULTIPLIER, MAX_MULTIPLIER,
    )

delta_f > 0  → actual winner scored higher on f → raise its weight
delta_f < 0  → chosen option scored higher on f but was wrong → lower its weight

Concurrency guarantee
---------------------
All public methods acquire _lock (threading.RLock) before touching _state or disk.
Writes go to a .tmp file first, then os.replace() does an atomic rename — so a
crash mid-write never leaves a half-written store file.

Offline / reconnect guarantee
------------------------------
_load() catches every IO and parse error and returns a fresh _FeedbackState.
The engine is therefore never blocked by a corrupt or missing store file.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Factor, OptionScore

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------
_LEARNING_RATE: float = 0.50   # step size per outcome
_MIN_MULTIPLIER: float = 0.25  # floor  — can't shrink below 25 % of original
_MAX_MULTIPLIER: float = 4.0   # ceiling — can't inflate past 4× original


@dataclass
class _FeedbackState:
    multipliers: dict[str, float] = field(default_factory=dict)
    total_outcomes: int = 0
    correct_outcomes: int = 0
    outcomes: list[dict[str, Any]] = field(default_factory=list)


class FeedbackStore:
    """
    Thread-safe, file-backed store for learned factor-weight multipliers.

    Thread safety
    -------------
    Every public method acquires self._lock before reading or writing _state.
    _save() writes to a .tmp file and uses os.replace() (atomic on POSIX and
    Windows NTFS) so concurrent writers never corrupt the store file.

    Resilience
    ----------
    _load() handles: missing file, empty file, truncated write, JSON syntax
    errors, and schema mismatches.  Any of these results in a clean slate —
    the engine keeps running as if the store never existed.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = self._load()
        logger.info("FeedbackStore initialised from '%s' (%d outcomes)",
                    self._path, self._state.total_outcomes)

    # ------------------------------------------------------------------
    # Evaluation-time hook
    # ------------------------------------------------------------------

    def get_adjusted_factors(self, factors: list[Factor]) -> list[Factor]:
        """Return Factors with weights scaled by learned multipliers (thread-safe)."""
        from dataclasses import replace
        with self._lock:
            result: list[Factor] = []
            for f in factors:
                m = self._state.multipliers.get(f.name, 1.0)
                result.append(f if m == 1.0 else replace(f, weight=round(f.weight * m, 6)))
            return result

    def applied_multipliers(self, factors: list[Factor]) -> dict[str, float]:
        """Return {factor_name: multiplier} for factors whose weight was adjusted."""
        with self._lock:
            return {
                f.name: round(self._state.multipliers[f.name], 4)
                for f in factors
                if f.name in self._state.multipliers
                and abs(self._state.multipliers[f.name] - 1.0) > 1e-9
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
        Record an outcome and update weight multipliers (thread-safe).

        Returns a dict describing what changed, suitable for JSON serialisation.
        """
        norm: dict[str, dict[str, float]] = {
            s.option_id: {fname: fc.normalized for fname, fc in s.by_factor.items()}
            for s in scores
        }

        with self._lock:
            self._state.total_outcomes += 1
            adjustments: dict[str, dict[str, float]] = {}

            if chosen_option == actual_winner or chosen_option is None:
                self._state.correct_outcomes += 1
                self._save()
                outcome = "correct" if chosen_option == actual_winner else "no_decision"
                logger.info("Feedback recorded: outcome=%s chosen=%s actual=%s",
                            outcome, chosen_option, actual_winner)
                return {"outcome": outcome, "adjustments": {}}

            # Wrong prediction — shift multipliers toward the actual winner.
            chosen_norm = norm.get(chosen_option, {})
            actual_norm = norm.get(actual_winner, {})
            all_factors = set(chosen_norm) | set(actual_norm)

            for fname in all_factors:
                self._state.multipliers.setdefault(fname, 1.0)
                before = self._state.multipliers[fname]
                delta = actual_norm.get(fname, 0.5) - chosen_norm.get(fname, 0.5)
                after = before + _LEARNING_RATE * delta
                after = max(_MIN_MULTIPLIER, min(_MAX_MULTIPLIER, after))
                self._state.multipliers[fname] = round(after, 6)
                if abs(after - before) > 1e-9:
                    adjustments[fname] = {"before": round(before, 4), "after": round(after, 4)}

            self._state.outcomes.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chosen": chosen_option,
                "actual": actual_winner,
                "factor_delta": {
                    fname: round(actual_norm.get(fname, 0.5) - chosen_norm.get(fname, 0.5), 4)
                    for fname in all_factors
                },
            })
            self._save()

        logger.info(
            "Feedback recorded: outcome=incorrect chosen=%s actual=%s adjustments=%s",
            chosen_option, actual_winner,
            {k: f"{v['before']:.3f}->{v['after']:.3f}" for k, v in adjustments.items()},
        )
        return {"outcome": "incorrect", "adjustments": adjustments}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total   = self._state.total_outcomes
            correct = self._state.correct_outcomes
            return {
                "total_outcomes_recorded": total,
                "correct_outcomes": correct,
                "accuracy": round(correct / total, 4) if total else None,
                "weight_multipliers": {
                    k: round(v, 4) for k, v in self._state.multipliers.items()
                },
                "interpretation": (
                    "multiplier > 1.0 → factor boosted by past feedback; "
                    "multiplier < 1.0 → factor attenuated."
                ),
            }

    # ------------------------------------------------------------------
    # Persistence — atomic write + resilient load
    # ------------------------------------------------------------------

    def _load(self) -> _FeedbackState:
        """
        Load store from disk.  Returns a clean _FeedbackState on any error so
        the engine is never blocked by a missing or corrupt file (offline-resilient).
        """
        if not self._path.exists():
            logger.info("No feedback store at '%s' — starting fresh.", self._path)
            return _FeedbackState()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            state = _FeedbackState(
                multipliers=raw.get("multipliers", {}),
                total_outcomes=raw.get("total_outcomes", 0),
                correct_outcomes=raw.get("correct_outcomes", 0),
                outcomes=raw.get("outcomes", []),
            )
            return state
        except Exception as exc:
            # Corrupt file, truncated write, schema mismatch — start fresh.
            logger.warning(
                "Could not load feedback store '%s' (%s: %s) — starting fresh.",
                self._path, type(exc).__name__, exc,
            )
            return _FeedbackState()

    def _save(self) -> None:
        """
        Atomic write: serialise to a .tmp file then os.replace() to the target.
        os.replace() is atomic on POSIX (rename syscall) and on Windows NTFS
        (MoveFileExW with MOVEFILE_REPLACE_EXISTING), so a crash mid-save never
        leaves a half-written store.
        """
        payload = {
            "multipliers": self._state.multipliers,
            "total_outcomes": self._state.total_outcomes,
            "correct_outcomes": self._state.correct_outcomes,
            "outcomes": self._state.outcomes,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)   # atomic rename
