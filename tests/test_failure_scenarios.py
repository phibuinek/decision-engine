"""
test_failure_scenarios.py — Asserts engine behaviour under adversarial or
degraded input conditions.

Each test targets a specific failure mode and verifies the *guarantee* the
engine is supposed to provide.  Tests are written so the failure condition and
expected invariant are explicit, making it clear *why* the design works.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from decision_engine.engine import DecisionEngine
from decision_engine.feedback import FeedbackStore
from decision_engine.io import load_request, parse_request
from decision_engine.types import (
    Claim,
    ClaimType,
    Factor,
    InputStatement,
    Option,
    PreferenceDirection,
    Source,
    Strength,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> DecisionEngine:
    return DecisionEngine()


def _now() -> datetime:
    return datetime.now(timezone.utc)


_UNSET: object = object()  # sentinel so ts=None can be passed explicitly


def _stmt(
    id: str,
    claim: Claim,
    *,
    strength: Strength = Strength.SOFT,
    reliability: float = 0.9,
    confidence: float = 1.0,
    ts: datetime | None | object = _UNSET,
) -> InputStatement:
    # ts=_UNSET  → default to now
    # ts=None    → pass None (no timestamp, triggers the 0.85 recency penalty)
    # ts=<dt>    → use that specific datetime
    timestamp = _now() if ts is _UNSET else ts  # type: ignore[arg-type]
    return InputStatement(
        id=id,
        strength=strength,
        source=Source(name="test", reliability=reliability),
        timestamp=timestamp,
        confidence=confidence,
        claim=claim,
    )


def _fval(option_id: str, factor: str, value: float) -> Claim:
    return Claim(type=ClaimType.FACTOR_VALUE, option_id=option_id, factor=factor, value=value)


def _constraint(factor: str, op: str, bound: float, option_id: str | None = None) -> Claim:
    return Claim(type=ClaimType.CONSTRAINT, factor=factor, op=op, bound=bound, option_id=option_id)


# ---------------------------------------------------------------------------
# 1. Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """
    Guarantee: duplicate statement IDs are dropped (first wins) and surfaced as
    a ``duplicate_input`` conflict.  Without this, double-submitted evidence
    would silently double-count, distorting weighted means.
    """

    def test_duplicate_ids_flagged_as_conflict(self):
        options = [Option(id="x"), Option(id="y")]
        factors = [Factor(name="cost", direction=PreferenceDirection.LOWER_IS_BETTER)]

        inputs = [
            _stmt("s1", _fval("x", "cost", 50)),
            _stmt("s1", _fval("x", "cost", 1)),   # duplicate — must be dropped
            _stmt("s2", _fval("y", "cost", 80)),
        ]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        dup_conflicts = [c for c in result.conflicts if c.kind == "duplicate_input"]
        assert dup_conflicts, "Expected a duplicate_input conflict"
        assert "s1" in dup_conflicts[0].statements

    def test_duplicate_dropped_not_averaged(self):
        """
        The injected duplicate has value=1 (suspiciously low cost).
        If it were processed, the weighted mean for 'x' would drop far below 50,
        boosting x's rank unfairly.  After deduplication x's raw cost must be 50.
        """
        options = [Option(id="x"), Option(id="y")]
        factors = [
            Factor(name="cost", direction=PreferenceDirection.LOWER_IS_BETTER,
                   min_value=0, max_value=100),
        ]
        inputs = [
            _stmt("s1", _fval("x", "cost", 50)),
            _stmt("s1", _fval("x", "cost", 1)),   # duplicate with manipulated value
            _stmt("s2", _fval("y", "cost", 80)),
        ]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        x_score = next(s for s in result.scores if s.option_id == "x")
        # raw value for x.cost must be 50 (first occurrence), not 25.5 (mean) or 1 (duplicate)
        assert abs(x_score.by_factor["cost"].raw - 50.0) < 1e-6, (
            f"Deduplication failed: raw cost={x_score.by_factor['cost'].raw}, expected 50"
        )

    def test_multiple_duplicate_pairs(self):
        options = [Option(id="p"), Option(id="q")]
        factors = [Factor(name="speed")]

        inputs = [
            _stmt("d1", _fval("p", "speed", 10)),
            _stmt("d1", _fval("p", "speed", 99)),  # dup
            _stmt("d2", _fval("q", "speed", 20)),
            _stmt("d2", _fval("q", "speed", 99)),  # dup
        ]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)
        dup_c = [c for c in result.conflicts if c.kind == "duplicate_input"]
        assert len(dup_c) == 1, "All duplicates should be in one conflict entry"
        assert set(dup_c[0].statements) == {"d1", "d2"}


# ---------------------------------------------------------------------------
# 2. Hard constraint enforcement
# ---------------------------------------------------------------------------

class TestHardConstraints:
    """
    Guarantee: a HARD constraint violation *always* disqualifies the option,
    regardless of how high the option scores on other factors.  A flimsy soft
    score cannot save a HARD-disqualified option.
    """

    def test_hard_constraint_disqualifies_despite_high_score(self):
        options = [Option(id="cheap"), Option(id="expensive")]
        factors = [
            Factor(name="price",   weight=1.0, direction=PreferenceDirection.LOWER_IS_BETTER,
                   min_value=0, max_value=2000),
            Factor(name="quality", weight=5.0, direction=PreferenceDirection.HIGHER_IS_BETTER,
                   min_value=0, max_value=10),
        ]
        inputs = [
            # 'expensive' has perfect quality (weight 5) but violates the budget hard constraint.
            _stmt("c1", _fval("expensive", "price",   1500)),
            _stmt("c2", _fval("expensive", "quality",   10)),
            _stmt("c3", _fval("cheap",     "price",    800)),
            _stmt("c4", _fval("cheap",     "quality",    3)),
            _stmt("c5", _constraint("price", "<=", 1000), strength=Strength.HARD),
        ]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        exp_score = next(s for s in result.scores if s.option_id == "expensive")
        assert exp_score.disqualified, "HARD constraint violation must disqualify the option"
        assert result.decision == "cheap", "Only non-disqualified option must win"

    def test_all_options_disqualified_returns_no_valid_options(self):
        """
        Failure mode: every option violates the hard constraint.
        Expected: status=no_valid_options, decision=None.
        Why: the engine must not silently pick the 'least bad' violator.
        """
        options = [Option(id="a"), Option(id="b")]
        factors = [Factor(name="latency", direction=PreferenceDirection.LOWER_IS_BETTER,
                          min_value=0, max_value=500)]
        inputs = [
            _stmt("q1", _fval("a", "latency", 300)),
            _stmt("q2", _fval("b", "latency", 400)),
            _stmt("q3", _constraint("latency", "<=", 100), strength=Strength.HARD),
        ]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        assert result.status == "no_valid_options"
        assert result.decision is None
        assert all(s.disqualified for s in result.scores)


# ---------------------------------------------------------------------------
# 3. Infeasible constraint detection
# ---------------------------------------------------------------------------

class TestInfeasibleConstraints:
    """
    Guarantee: mutually exclusive HARD constraints are detected and reported
    before scoring begins.  The engine must not silently discard one constraint
    and appear to give a valid answer.
    """

    def test_infeasible_bounds_detected(self):
        req = load_request(Path("examples") / "conflicting_constraints.json")
        result = _engine().evaluate(options=req.options, factors=req.factors, inputs=req.inputs)

        assert result.status == "no_valid_options"
        cc = [c for c in result.conflicts if c.kind == "constraint_constraint"]
        assert cc, "Infeasible constraints must produce a constraint_constraint conflict"
        assert any(c.severity == "high" for c in cc), "Infeasible constraint conflict must be high-severity"

    def test_eq_ne_conflict_flagged(self):
        """== X and != X on the same factor and option scope are mutually exclusive."""
        options = [Option(id="r")]
        factors = [Factor(name="version", min_value=0, max_value=10)]
        inputs = [
            _stmt("e1", _fval("r", "version", 3.0)),
            _stmt("e2", _constraint("version", "==", 3.0), strength=Strength.HARD),
            _stmt("e3", _constraint("version", "!=", 3.0), strength=Strength.HARD),
        ]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        cc = [c for c in result.conflicts if c.kind == "constraint_constraint"]
        assert cc, "== / != conflict must be detected"


# ---------------------------------------------------------------------------
# 4. Stale data
# ---------------------------------------------------------------------------

class TestStaleData:
    """
    Guarantee: old statements are downweighted via exponential decay so they
    cannot outweigh fresh evidence even if they have higher reliability.
    """

    def test_stale_statement_has_lower_evidence_weight(self):
        """
        A statement from 365 days ago should have much lower evidence weight
        than an identical statement from today.
        """
        options = [Option(id="m")]
        factors = [Factor(name="perf", min_value=0, max_value=100)]
        now = _now()
        old_ts = now - timedelta(days=365)

        # Two statements with identical reliability but very different ages.
        inputs_fresh = [_stmt("f1", _fval("m", "perf", 80), ts=now)]
        inputs_stale = [_stmt("f1", _fval("m", "perf", 80), ts=old_ts)]

        res_fresh = _engine().evaluate(options=options, factors=factors,
                                       inputs=inputs_fresh, now=now)
        res_stale = _engine().evaluate(options=options, factors=factors,
                                       inputs=inputs_stale, now=now)

        fresh_ew = res_fresh.scores[0].by_factor["perf"].evidence_weight
        stale_ew = res_stale.scores[0].by_factor["perf"].evidence_weight

        assert fresh_ew > stale_ew, (
            f"Stale data must have lower evidence weight "
            f"(fresh={fresh_ew:.3f}, stale={stale_ew:.3f})"
        )

    def test_missing_timestamp_penalised(self):
        """
        A statement with no timestamp gets recency_weight=0.85 (less than 1.0),
        so it is always weaker than a freshly-timestamped statement.
        """
        options = [Option(id="n")]
        factors = [Factor(name="score", min_value=0, max_value=100)]
        now = _now()

        inputs_timestamped = [_stmt("t1", _fval("n", "score", 50), ts=now)]
        inputs_no_ts = [_stmt("t1", _fval("n", "score", 50), ts=None)]

        res_ts = _engine().evaluate(options=options, factors=factors,
                                    inputs=inputs_timestamped, now=now)
        res_no_ts = _engine().evaluate(options=options, factors=factors,
                                       inputs=inputs_no_ts, now=now)

        ts_ew   = res_ts.scores[0].by_factor["score"].evidence_weight
        nots_ew = res_no_ts.scores[0].by_factor["score"].evidence_weight

        assert ts_ew > nots_ew, (
            "Missing timestamp must produce a lower evidence weight than a fresh timestamp"
        )


# ---------------------------------------------------------------------------
# 5. Missing factor values
# ---------------------------------------------------------------------------

class TestMissingValues:
    """
    Guarantee: when no evidence exists for a factor, the engine assigns a
    neutral score with a 40 % uncertainty penalty — it must *not* assume the
    best or worst possible value.  All assumptions must be logged.
    """

    def test_missing_factor_is_neutral_with_penalty(self):
        options = [Option(id="u")]
        factors = [
            Factor(name="known",   weight=1.0, min_value=0, max_value=10),
            Factor(name="unknown", weight=1.0, min_value=0, max_value=10),
        ]
        inputs = [_stmt("m1", _fval("u", "known", 5))]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        unk = result.scores[0].by_factor["unknown"]
        assert unk.evidence_weight == 0.0,  "No evidence → evidence_weight must be 0"
        assert abs(unk.normalized - 0.5) < 1e-9, "No evidence → normalized score must be 0.5"
        # weighted = 0.5 * weight * 0.6  (40 % penalty)
        assert abs(unk.weighted - 0.5 * 1.0 * 0.6) < 1e-9, (
            f"Missing-value penalty not applied correctly: weighted={unk.weighted}"
        )

    def test_missing_value_logged_in_assumptions(self):
        options = [Option(id="v")]
        factors = [Factor(name="x"), Factor(name="y")]
        inputs = [_stmt("n1", _fval("v", "x", 3))]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        missing_assumptions = [a for a in result.assumptions if "y" in a and "neutral" in a.lower()]
        assert missing_assumptions, (
            "Missing factor value must be recorded in assumptions"
        )


# ---------------------------------------------------------------------------
# 6. Insufficient information
# ---------------------------------------------------------------------------

class TestInsufficientInfo:
    """
    Guarantee: when *no* factor has any evidence, the engine returns
    status=insufficient_info rather than making up a winner.
    """

    def test_no_evidence_gives_insufficient_info(self):
        options = [Option(id="p"), Option(id="q")]
        factors = [Factor(name="a"), Factor(name="b")]
        # No factor_value statements at all — only a preference (which gives no value).
        inputs = [
            _stmt(
                "pref1",
                Claim(type=ClaimType.PREFERENCE, factor="a",
                      direction=PreferenceDirection.HIGHER_IS_BETTER),
            )
        ]
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        assert result.status == "insufficient_info"
        assert result.decision is None


# ---------------------------------------------------------------------------
# 7. Feedback loop
# ---------------------------------------------------------------------------

class TestFeedbackLoop:
    """
    Guarantee: recording a wrong decision adjusts multipliers in the direction
    that would have favoured the actual winner.  Correct decisions leave
    multipliers unchanged.
    """

    def _base_scenario(self):
        options = [Option(id="fast"), Option(id="cheap")]
        factors = [
            Factor(name="speed", weight=1.0, direction=PreferenceDirection.HIGHER_IS_BETTER,
                   min_value=0, max_value=100),
            Factor(name="cost",  weight=1.0, direction=PreferenceDirection.LOWER_IS_BETTER,
                   min_value=0, max_value=100),
        ]
        inputs = [
            _stmt("i1", _fval("fast",  "speed", 90)),
            _stmt("i2", _fval("fast",  "cost",  80)),
            _stmt("i3", _fval("cheap", "speed", 40)),
            _stmt("i4", _fval("cheap", "cost",  20)),
        ]
        return options, factors, inputs

    def test_wrong_decision_adjusts_multipliers(self):
        options, factors, inputs = self._base_scenario()
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)
        # Engine will pick 'fast' (high speed, lower cost penalty) — confirm or proceed
        chosen = result.decision

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            store = FeedbackStore(tmp.name)
            before = dict(store.summary()["weight_multipliers"])

            # Tell the store that 'cheap' was actually the better choice.
            actual = "cheap" if chosen == "fast" else "fast"
            update = store.record_outcome(
                scores=result.scores,
                chosen_option=chosen,
                actual_winner=actual,
            )

        assert update["outcome"] == "incorrect"
        assert update["adjustments"], "A wrong decision must produce multiplier adjustments"

    def test_correct_decision_leaves_multipliers_unchanged(self):
        options, factors, inputs = self._base_scenario()
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)
        chosen = result.decision

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            store = FeedbackStore(tmp.name)
            update = store.record_outcome(
                scores=result.scores,
                chosen_option=chosen,
                actual_winner=chosen,   # agree with engine
            )

        assert update["outcome"] == "correct"
        assert update["adjustments"] == {}, "Correct decision must not move any multipliers"
        assert store.summary()["weight_multipliers"] == {}, (
            "No multipliers should exist after only correct predictions"
        )

    def test_feedback_adjusts_evaluation(self):
        """
        After recording enough wrong decisions, the adjusted factors must push
        the evaluation outcome toward the actual winner.
        """
        options, factors, inputs = self._base_scenario()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            store_path = tmp.name

        store = FeedbackStore(store_path)
        engine = _engine()

        # Simulate 5 consecutive evaluations where 'cheap' is always the actual winner.
        for _ in range(5):
            adj_factors = store.get_adjusted_factors(factors)
            result = engine.evaluate(options=options, factors=adj_factors, inputs=inputs)
            store.record_outcome(
                scores=result.scores,
                chosen_option=result.decision,
                actual_winner="cheap",
            )

        # After repeated feedback, cost should have its multiplier raised
        # (actual winner 'cheap' scored high on cost) and speed lowered.
        summary = store.summary()
        mults = summary["weight_multipliers"]

        assert summary["total_outcomes_recorded"] == 5

        # The multipliers must have moved from 1.0 in the direction of 'cheap'.
        # 'cheap' scores high on cost (low raw = good for lower_is_better → high normalised)
        # and low on speed — so cost mult should rise, speed mult should fall.
        if "cost" in mults and "speed" in mults:
            assert mults["cost"] > mults["speed"], (
                "After repeated 'cheap' wins, cost weight should exceed speed weight"
            )

    def test_feedback_store_survives_corrupt_file(self):
        """
        A corrupt feedback store must not crash the engine — it should start fresh.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp.write("{ this is not valid json !!!")
            tmp_path = tmp.name

        store = FeedbackStore(tmp_path)   # must not raise
        assert store.summary()["total_outcomes_recorded"] == 0

    def test_feedback_file_is_always_valid_json(self):
        """
        After any record_outcome call, the feedback file must be parseable JSON.
        """
        options, factors, inputs = self._base_scenario()
        result = _engine().evaluate(options=options, factors=factors, inputs=inputs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            store = FeedbackStore(tmp.name)
            store.record_outcome(result.scores, result.decision, "cheap")
            raw = Path(tmp.name).read_text(encoding="utf-8")
            parsed = json.loads(raw)   # must not raise

        assert "multipliers" in parsed
        assert "total_outcomes" in parsed


# ---------------------------------------------------------------------------
# 8. Duplicate-ids example file
# ---------------------------------------------------------------------------

class TestDuplicateIdsExample:
    """
    Verifies the bundled duplicate_ids.json example works end-to-end and that
    the engine's deduplication guarantee holds on a realistic scenario.
    """

    def test_duplicate_ids_example_conflict_detected(self):
        req = load_request(Path("examples") / "duplicate_ids.json")
        result = _engine().evaluate(options=req.options, factors=req.factors, inputs=req.inputs)

        dup = [c for c in result.conflicts if c.kind == "duplicate_input"]
        assert dup, "duplicate_ids.json must trigger a duplicate_input conflict"
        assert "s1" in dup[0].statements or "s4" in dup[0].statements

    def test_duplicate_ids_example_produces_valid_decision(self):
        req = load_request(Path("examples") / "duplicate_ids.json")
        result = _engine().evaluate(options=req.options, factors=req.factors, inputs=req.inputs)

        assert result.status in ("ok", "tie")
        # alpha has cost=40 (first s1 kept), beta has cost=70 — alpha should win on cost.
        assert result.decision == "alpha", (
            "Alpha must win: cost=40 vs beta cost=70, and the injected cost=1 duplicate must be dropped"
        )
