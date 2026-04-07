from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .types import (
    ClaimType,
    Conflict,
    DecisionResult,
    Factor,
    FactorContribution,
    InputStatement,
    Option,
    OptionScore,
    PreferenceDirection,
    Strength,
)


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _recency_weight(ts: datetime | None, now: datetime) -> float:
    """
    Smoothly downweights stale statements.
    - If timestamp is missing, default to 0.85 (slight penalty, still usable).
    - Half-life ≈ 90 days.
    """
    if ts is None:
        return 0.85
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    half_life = 90.0
    return 0.5 ** (age_days / half_life)


def _evidence_weight(stmt: InputStatement, now: datetime) -> float:
    return _clamp01(stmt.source.reliability) * _clamp01(stmt.confidence) * _recency_weight(
        stmt.timestamp, now
    )


class DecisionEngine:
    """
    Deterministic decision engine with:
    - Conflict detection (values, constraints, preferences)
    - Multi-factor scoring with evidence aggregation
    - Explainable trace output
    """

    def evaluate(
        self,
        *,
        options: list[Option],
        factors: list[Factor],
        inputs: list[InputStatement],
        now: datetime | None = None,
        tie_epsilon: float = 1e-6,
    ) -> DecisionResult:
        now = now or _now_utc()

        factor_by_name = {f.name: f for f in factors}
        option_by_id = {o.id: o for o in options}

        assumptions: list[str] = []
        conflicts: list[Conflict] = []

        # Collect evidence per option+factor for factor_value claims.
        value_evidence: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
        # Collect constraints per option+factor.
        # Stored as (op, bound, evidence_weight, statement_id, strength)
        constraints: dict[tuple[str, str], list[tuple[str, float, float, str, Strength]]] = {}
        # Preference statements per factor: (direction, evidence_weight, statement_id)
        pref_evidence: dict[str, list[tuple[PreferenceDirection, float, str]]] = {}

        unknowns: list[str] = []
        for stmt in inputs:
            c = stmt.claim
            if c.type == ClaimType.FACTOR_VALUE:
                if not c.factor or not c.option_id or c.value is None:
                    unknowns.append(stmt.id)
                    continue
                if c.factor not in factor_by_name or c.option_id not in option_by_id:
                    unknowns.append(stmt.id)
                    continue
                ew = _evidence_weight(stmt, now)
                key = (c.option_id, c.factor)
                value_evidence.setdefault(key, []).append((c.value, ew, stmt.id))
            elif c.type == ClaimType.CONSTRAINT:
                if not c.factor or c.bound is None or not c.op:
                    unknowns.append(stmt.id)
                    continue
                # Constraint can be option-specific or global (option_id None).
                if c.factor not in factor_by_name:
                    unknowns.append(stmt.id)
                    continue
                ew = _evidence_weight(stmt, now)
                key = (c.option_id or "*", c.factor)
                constraints.setdefault(key, []).append((c.op, c.bound, ew, stmt.id, stmt.strength))
            elif c.type == ClaimType.PREFERENCE:
                if not c.factor:
                    unknowns.append(stmt.id)
                    continue
                if c.factor not in factor_by_name:
                    unknowns.append(stmt.id)
                    continue
                if c.direction is not None:
                    ew = _evidence_weight(stmt, now)
                    pref_evidence.setdefault(c.factor, []).append((c.direction, ew, stmt.id))
            else:
                unknowns.append(stmt.id)

        if unknowns:
            assumptions.append(
                f"Ignored {len(unknowns)} malformed/unknown statements: {', '.join(unknowns)}"
            )

        # Resolve preference overrides; detect conflicts if mixed directions exist.
        pref_overrides: dict[str, PreferenceDirection] = {}
        for factor, prefs in pref_evidence.items():
            dirs = {d for (d, _ew, _sid) in prefs}
            if len(dirs) == 1:
                pref_overrides[factor] = next(iter(dirs))
                continue
            # Conflict: choose direction with higher total evidence weight.
            totals: dict[PreferenceDirection, float] = {}
            for (d, ew, _sid) in prefs:
                totals[d] = totals.get(d, 0.0) + ew
            chosen = max(totals.items(), key=lambda kv: kv[1])[0]
            pref_overrides[factor] = chosen
            conflicts.append(
                Conflict(
                    kind="preference_preference",
                    factor=factor,
                    option_id=None,
                    statements=tuple(sid for (_d, _ew, sid) in prefs),
                    summary=f"Conflicting preferences for '{factor}' ({', '.join(sorted(x.value for x in dirs))}).",
                    severity="medium",
                    resolution=f"Chose '{chosen.value}' by higher total evidence weight.",
                )
            )

        # Detect conflicts in factor value evidence.
        for (opt_id, factor), evs in value_evidence.items():
            if len(evs) < 2:
                continue
            values = [v for (v, _ew, _sid) in evs]
            vmin, vmax = min(values), max(values)
            if vmax - vmin <= 0:
                continue
            # If values differ materially, register conflict and resolve by weighted mean.
            total_w = sum(w for (_v, w, _sid) in evs)
            if total_w <= 0:
                continue
            mean = sum(v * w for (v, w, _sid) in evs) / total_w
            conflicts.append(
                Conflict(
                    kind="value_value",
                    factor=factor,
                    option_id=opt_id,
                    statements=tuple(sid for (_v, _w, sid) in evs),
                    summary=f"Conflicting values for {factor} on {opt_id}: range [{vmin}, {vmax}]",
                    severity="medium" if (vmax - vmin) / (abs(mean) + 1e-9) < 0.5 else "high",
                    resolution="Aggregated using evidence-weighted mean (reliability × confidence × recency).",
                )
            )

        # Detect conflicts between constraints (same option scope).
        for (opt_scope, factor), cons in constraints.items():
            if len(cons) < 2:
                continue
            # Pragmatic feasibility checks for HARD constraints.
            hard = [c for c in cons if c[4] == Strength.HARD]
            if len(hard) < 2:
                continue

            eq = [c for c in hard if c[0] == "=="]
            ne = [c for c in hard if c[0] == "!="]
            if eq:
                b = eq[0][1]
                if any(abs(c[1] - b) < 1e-9 for c in ne):
                    conflicts.append(
                        Conflict(
                            kind="constraint_constraint",
                            factor=factor,
                            option_id=None if opt_scope == "*" else opt_scope,
                            statements=tuple([c[3] for c in hard]),
                            summary=f"Hard constraint conflict on {factor}: '==' and '!=' for {b}",
                            severity="high",
                            resolution="Constraints appear mutually exclusive; options may be disqualified conservatively.",
                        )
                    )

            # Interval infeasibility: max(lower_bound) > min(upper_bound)
            lowers = []
            uppers = []
            for (op, bound, _ew, sid, _strength) in hard:
                if op in (">", ">="):
                    lowers.append((bound, op, sid))
                elif op in ("<", "<="):
                    uppers.append((bound, op, sid))
            if lowers and uppers:
                max_lower = max(lowers, key=lambda x: x[0])
                min_upper = min(uppers, key=lambda x: x[0])
                if max_lower[0] > min_upper[0]:
                    conflicts.append(
                        Conflict(
                            kind="constraint_constraint",
                            factor=factor,
                            option_id=None if opt_scope == "*" else opt_scope,
                            statements=tuple([c[3] for c in hard]),
                            summary=(
                                f"Infeasible hard constraints on {factor}: "
                                f"{max_lower[1]} {max_lower[0]} (from {max_lower[2]}) "
                                f"conflicts with {min_upper[1]} {min_upper[0]} (from {min_upper[2]})."
                            ),
                            severity="high",
                            resolution="Marked as a high-severity conflict; constraint enforcement remains conservative.",
                        )
                    )

        # Aggregate final value per option+factor.
        agg_values: dict[tuple[str, str], tuple[float, float, list[str]]] = {}
        for key, evs in value_evidence.items():
            total_w = sum(w for (_v, w, _sid) in evs)
            if total_w <= 0:
                continue
            mean = sum(v * w for (v, w, _sid) in evs) / total_w
            agg_values[key] = (mean, total_w, [sid for (_v, _w, sid) in evs])

        # Infer normalization ranges if missing.
        inferred_ranges: dict[str, tuple[float, float]] = {}
        for f in factors:
            if f.min_value is not None and f.max_value is not None:
                inferred_ranges[f.name] = (f.min_value, f.max_value)
                continue
            observed = [
                v
                for ((opt_id, factor), (v, _tw, _sids)) in agg_values.items()
                if factor == f.name
            ]
            if len(observed) >= 2:
                inferred_ranges[f.name] = (min(observed), max(observed))
            elif len(observed) == 1:
                # Degenerate range; avoid division by zero.
                inferred_ranges[f.name] = (observed[0] - 1.0, observed[0] + 1.0)
                assumptions.append(
                    f"Factor '{f.name}' had only one observed value; used ±1 range for normalization."
                )
            else:
                inferred_ranges[f.name] = (0.0, 1.0)
                assumptions.append(
                    f"Factor '{f.name}' had no observed values; defaulted normalization range to [0, 1]."
                )

        def normalize(factor: Factor, raw: float) -> float:
            lo, hi = inferred_ranges[factor.name]
            if hi - lo == 0:
                return 0.5
            x = (raw - lo) / (hi - lo)
            x = 0.0 if x < 0 else 1.0 if x > 1 else x
            direction = pref_overrides.get(factor.name, factor.direction)
            return x if direction == PreferenceDirection.HIGHER_IS_BETTER else 1.0 - x

        # Apply constraints:
        # - HARD constraints disqualify when violated.
        # - SOFT constraints add a penalty (do not disqualify).
        # Constraints apply either globally (option "*") or option-specific.
        def violates(op: str, value: float, bound: float) -> bool:
            if op == "<=":
                return not (value <= bound)
            if op == "<":
                return not (value < bound)
            if op == ">=":
                return not (value >= bound)
            if op == ">":
                return not (value > bound)
            if op == "==":
                return not (abs(value - bound) < 1e-9)
            if op == "!=":
                return abs(value - bound) < 1e-9
            return False

        scores: list[OptionScore] = []
        for opt in options:
            disqualified = False
            disqualify_reasons: list[str] = []
            by_factor: dict[str, FactorContribution] = {}
            total = 0.0

            for f in factors:
                key = (opt.id, f.name)
                if key in agg_values:
                    raw, total_ev_w, sids = agg_values[key]
                    norm = normalize(f, raw)
                    weighted = norm * f.weight
                    # Penalize weak evidence slightly so "flimsy" info doesn't dominate.
                    evidence_multiplier = 0.75 + 0.25 * _clamp01(total_ev_w)
                    weighted *= evidence_multiplier
                    contrib = FactorContribution(
                        factor=f.name,
                        raw=raw,
                        normalized=norm,
                        weighted=weighted,
                        evidence_weight=_clamp01(total_ev_w),
                        notes=[f"Statements: {', '.join(sids)}"],
                    )
                else:
                    # Missing factor value: treat as neutral but add uncertainty penalty.
                    assumptions.append(
                        f"Missing value for factor '{f.name}' on option '{opt.id}'; treated as neutral."
                    )
                    contrib = FactorContribution(
                        factor=f.name,
                        raw=0.0,
                        normalized=0.5,
                        weighted=0.5 * f.weight * 0.6,  # mild penalty vs fully-known neutral
                        evidence_weight=0.0,
                        notes=["No evidence; applied uncertainty penalty."],
                    )

                # Apply constraints for this factor if any.
                applied_constraints = constraints.get(("*", f.name), []) + constraints.get(
                    (opt.id, f.name), []
                )
                if applied_constraints:
                    # If raw is missing, we can't verify; treat as soft disqualifier (uncertainty).
                    if key not in agg_values:
                        disqualify_reasons.append(
                            f"Constraint on '{f.name}' could not be verified due to missing value."
                        )
                        # extra penalty
                        contrib.weighted *= 0.7
                    else:
                        raw_val = agg_values[key][0]
                        for (op, bound, _ew, sid, strength) in applied_constraints:
                            if violates(op, raw_val, bound):
                                if strength == Strength.HARD:
                                    disqualified = True
                                    disqualify_reasons.append(
                                        f"Violated hard constraint {f.name} {op} {bound} (from {sid})."
                                    )
                                else:
                                    # Soft constraint: keep option, but penalize this factor.
                                    contrib.weighted *= 0.75
                                    contrib.notes.append(
                                        f"Violated soft constraint {f.name} {op} {bound} (from {sid})."
                                    )
                by_factor[f.name] = contrib
                total += contrib.weighted

            scores.append(
                OptionScore(
                    option_id=opt.id,
                    total=total,
                    by_factor=by_factor,
                    disqualified=disqualified,
                    disqualify_reasons=disqualify_reasons,
                )
            )

        valid = [s for s in scores if not s.disqualified]
        if not valid:
            status = "no_valid_options"
            decision = None
        else:
            valid_sorted = sorted(valid, key=lambda s: s.total, reverse=True)
            best = valid_sorted[0]
            # Tie check among valid.
            tied = [s for s in valid_sorted if abs(s.total - best.total) <= tie_epsilon]
            if len(tied) > 1:
                status = "tie"
                decision = None
            else:
                status = "ok"
                decision = best.option_id

        # If everything is neutral due to missing info, mark insufficient.
        if all(all(fc.evidence_weight <= 0 for fc in s.by_factor.values()) for s in scores):
            status = "insufficient_info"
            decision = None

        explanation = self._build_explanation(
            options=options,
            factors=factors,
            scores=scores,
            decision=decision,
            status=status,
            conflicts=conflicts,
            assumptions=assumptions,
            now=now,
            pref_overrides=pref_overrides,
            inferred_ranges=inferred_ranges,
        )

        return DecisionResult(
            decision=decision,
            status=status,
            scores=sorted(scores, key=lambda s: s.total, reverse=True),
            conflicts=conflicts,
            assumptions=_dedupe_preserve_order(assumptions),
            explanation=explanation,
        )

    def _build_explanation(
        self,
        *,
        options: list[Option],
        factors: list[Factor],
        scores: list[OptionScore],
        decision: str | None,
        status: str,
        conflicts: list[Conflict],
        assumptions: list[str],
        now: datetime,
        pref_overrides: dict[str, PreferenceDirection],
        inferred_ranges: dict[str, tuple[float, float]],
    ) -> dict[str, Any]:
        option_labels = {o.id: (o.label or o.id) for o in options}
        factor_meta = []
        for f in factors:
            direction = pref_overrides.get(f.name, f.direction).value
            lo, hi = inferred_ranges.get(f.name, (None, None))
            factor_meta.append(
                {
                    "name": f.name,
                    "weight": f.weight,
                    "direction": direction,
                    "normalization_range": [lo, hi],
                }
            )

        # A compact human summary
        sorted_scores = sorted(scores, key=lambda s: s.total, reverse=True)
        top3 = sorted_scores[:3]
        leaderboard = [
            {
                "option_id": s.option_id,
                "label": option_labels.get(s.option_id, s.option_id),
                "total": s.total,
                "disqualified": s.disqualified,
            }
            for s in top3
        ]

        if status == "ok" and decision:
            summary = (
                f"Selected '{option_labels.get(decision, decision)}' because it achieved the highest "
                f"multi-factor score among non-disqualified options."
            )
        elif status == "tie":
            summary = "No single best option: top options are tied within the configured epsilon."
        elif status == "no_valid_options":
            summary = "All options were disqualified by constraints."
        else:
            summary = "Insufficient reliable information to choose an option confidently."

        return {
            "summary": summary,
            "now": now.isoformat(),
            "leaderboard": leaderboard,
            "decision": decision,
            "status": status,
            "factors": factor_meta,
            "conflicts": [asdict(c) for c in conflicts],
            "assumptions": _dedupe_preserve_order(assumptions),
            "trace": {
                "scores": [
                    {
                        "option_id": s.option_id,
                        "total": s.total,
                        "disqualified": s.disqualified,
                        "disqualify_reasons": s.disqualify_reasons,
                        "by_factor": {
                            k: asdict(v) for (k, v) in sorted(s.by_factor.items(), key=lambda kv: kv[0])
                        },
                    }
                    for s in sorted_scores
                ]
            },
        }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

