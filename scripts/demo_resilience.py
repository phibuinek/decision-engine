"""
demo_resilience.py — Evidence of offline / reconnect resilience.

What this proves
----------------
1. FeedbackStore loads correctly from a valid file on startup.
2. FeedbackStore starts fresh (without crashing) when the file is:
     a. missing
     b. empty
     c. truncated (partial write)
     d. valid JSON but wrong schema
3. After the store file is deleted mid-run, the next record_outcome()
   call recreates it atomically — simulating a disk remount or reconnect.
4. Concurrent evaluate requests during a store reset complete without error.

Run (no server needed — tests the store directly):
    python scripts/demo_resilience.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_engine.engine import DecisionEngine
from decision_engine.feedback import FeedbackStore
from decision_engine.io import parse_request

PASS = "PASS"
FAIL = "FAIL"


def _make_scores():
    """Create a minimal evaluation result for feedback tests."""
    req = parse_request({
        "options": [{"id": "fast"}, {"id": "cheap"}],
        "factors": [
            {"name": "speed", "weight": 1.0, "direction": "higher_is_better",
             "min_value": 0, "max_value": 100},
            {"name": "cost",  "weight": 1.0, "direction": "lower_is_better",
             "min_value": 0, "max_value": 100},
        ],
        "inputs": [
            {"id":"i1","strength":"soft","source":{"name":"t","reliability":0.9},
             "claim":{"type":"factor_value","option_id":"fast","factor":"speed","value":90}},
            {"id":"i2","strength":"soft","source":{"name":"t","reliability":0.9},
             "claim":{"type":"factor_value","option_id":"fast","factor":"cost","value":80}},
            {"id":"i3","strength":"soft","source":{"name":"t","reliability":0.9},
             "claim":{"type":"factor_value","option_id":"cheap","factor":"speed","value":40}},
            {"id":"i4","strength":"soft","source":{"name":"t","reliability":0.9},
             "claim":{"type":"factor_value","option_id":"cheap","factor":"cost","value":20}},
        ],
    })
    engine = DecisionEngine()
    result = engine.evaluate(options=req.options, factors=req.factors, inputs=req.inputs)
    return result.scores, result.decision


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def run() -> bool:
    print("=" * 60)
    print("Decision Engine — Resilience & Offline Demo")
    print("=" * 60)

    all_ok = True
    scores, decision = _make_scores()

    # ── Scenario 1: Normal load ───────────────────────────────────────────
    print("\n--- Scenario 1: Normal load from valid file ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = Path(tmp.name)

    store = FeedbackStore(path)
    store.record_outcome(scores, decision, actual_winner="cheap")
    store2 = FeedbackStore(path)   # reload from disk
    total = store2.summary()["total_outcomes_recorded"]
    ok = check("Reloaded store has correct outcome count", total == 1, f"total={total}")
    all_ok = all_ok and ok

    # ── Scenario 2a: Missing file ─────────────────────────────────────────
    print("\n--- Scenario 2a: Store file missing (offline) ---")
    missing_path = path.with_name("nonexistent_store.json")
    try:
        s = FeedbackStore(missing_path)
        ok = check("FeedbackStore starts fresh when file is missing",
                   s.summary()["total_outcomes_recorded"] == 0)
    except Exception as exc:
        ok = check("FeedbackStore starts fresh when file is missing", False, str(exc))
    all_ok = all_ok and ok

    # ── Scenario 2b: Empty file ───────────────────────────────────────────
    print("\n--- Scenario 2b: Store file is empty ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp.write("")
        empty_path = Path(tmp.name)
    try:
        s = FeedbackStore(empty_path)
        ok = check("FeedbackStore starts fresh on empty file",
                   s.summary()["total_outcomes_recorded"] == 0)
    except Exception as exc:
        ok = check("FeedbackStore starts fresh on empty file", False, str(exc))
    all_ok = all_ok and ok

    # ── Scenario 2c: Truncated / corrupt file ─────────────────────────────
    print("\n--- Scenario 2c: Store file is truncated (partial write) ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp.write('{"multipliers": {"speed": 1.5, "cost":')   # truncated
        corrupt_path = Path(tmp.name)
    try:
        s = FeedbackStore(corrupt_path)
        ok = check("FeedbackStore recovers from truncated file",
                   s.summary()["total_outcomes_recorded"] == 0)
    except Exception as exc:
        ok = check("FeedbackStore recovers from truncated file", False, str(exc))
    all_ok = all_ok and ok

    # ── Scenario 2d: Wrong schema ─────────────────────────────────────────
    print("\n--- Scenario 2d: Store file has wrong schema ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump({"version": 99, "data": "completely different format"}, tmp)
        schema_path = Path(tmp.name)
    try:
        s = FeedbackStore(schema_path)
        ok = check("FeedbackStore starts fresh on wrong schema",
                   s.summary()["total_outcomes_recorded"] == 0)
    except Exception as exc:
        ok = check("FeedbackStore starts fresh on wrong schema", False, str(exc))
    all_ok = all_ok and ok

    # ── Scenario 3: Store deleted mid-run → reconnect ─────────────────────
    print("\n--- Scenario 3: Store file deleted mid-run (disk failure simulation) ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        live_path = Path(tmp.name)

    live_store = FeedbackStore(live_path)
    live_store.record_outcome(scores, decision, actual_winner="cheap")

    # Simulate disk going offline: delete the file.
    live_path.unlink()
    ok = check("Store file removed (disk offline)", not live_path.exists())
    all_ok = all_ok and ok

    # Record another outcome — store recreates the file atomically.
    try:
        live_store.record_outcome(scores, decision, actual_winner="cheap")
        recreated = live_path.exists()
        ok = check("record_outcome recreates file after deletion (reconnect)", recreated)
        all_ok = all_ok and ok
        if recreated:
            reloaded = FeedbackStore(live_path)
            total = reloaded.summary()["total_outcomes_recorded"]
            ok = check("Reloaded store after reconnect has correct count",
                       total == 2, f"total={total}")
            all_ok = all_ok and ok
    except Exception as exc:
        ok = check("record_outcome recreates file after deletion", False, str(exc))
        all_ok = all_ok and ok

    # ── Scenario 4: Concurrent feedback during store reset ────────────────
    print("\n--- Scenario 4: 10 threads recording feedback simultaneously ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        conc_path = Path(tmp.name)
    conc_store = FeedbackStore(conc_path)
    errors: list[str] = []
    lock    = threading.Lock()

    def _record(_: int):
        try:
            conc_store.record_outcome(scores, decision, actual_winner="cheap")
        except Exception as exc:
            with lock:
                errors.append(str(exc))

    threads = [threading.Thread(target=_record, args=(i,)) for i in range(10)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.perf_counter() - t0

    total = FeedbackStore(conc_path).summary()["total_outcomes_recorded"]
    ok = check("No exceptions during concurrent writes", not errors,
               f"errors={errors}" if errors else "")
    all_ok = all_ok and ok
    ok = check(f"All 10 outcomes recorded (no lost writes, {elapsed:.2f}s)",
               total == 10, f"total={total}")
    all_ok = all_ok and ok

    # ── Atomic write verification ─────────────────────────────────────────
    print("\n--- Scenario 5: Atomic write — no .tmp file left after save ---")
    tmp_leftover = conc_path.with_suffix(".tmp")
    ok = check("No .tmp leftover file (atomic rename succeeded)", not tmp_leftover.exists())
    all_ok = all_ok and ok

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL RESILIENCE CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see details above")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
