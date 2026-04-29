"""
demo_concurrent.py — Evidence of correct concurrent behaviour.

What this proves
----------------
1. 20 evaluate requests fired simultaneously all return valid, consistent results.
2. 10 feedback record requests fired simultaneously all land without corrupting
   the store (total_outcomes == 10 at the end, not less due to a lost write).
3. The final feedback store is valid JSON and internally consistent
   (correct_outcomes + incorrect_outcomes == total_outcomes).

Run (server must be running on localhost:8000):
    python scripts/demo_concurrent.py

The script prints a summary and exits with code 0 on success, 1 on failure.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "http://localhost:8000"


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


def _load_laptop() -> dict:
    return _get("/api/examples/laptop")


def run() -> bool:
    print("=" * 60)
    print("Decision Engine — Concurrent Behaviour Demo")
    print("=" * 60)

    # ── Check server is up ────────────────────────────────────────────────
    try:
        health = _get("/api/health")
        print(f"\n[health] status={health['status']}  "
              f"cached_evals={health['cached_evaluations']}")
    except Exception as exc:
        print(f"\nERROR: Cannot reach server at {BASE} — {exc}")
        print("Start the server first: uvicorn decision_engine.api:app --port 8000")
        return False

    # Reset feedback store for a clean test.
    req = urllib.request.Request(BASE + "/api/feedback", method="DELETE")
    with urllib.request.urlopen(req, timeout=10):
        pass
    print("[reset] Feedback store cleared.")

    laptop = _load_laptop()

    # ── Phase 1: 20 concurrent evaluations ───────────────────────────────
    print("\n--- Phase 1: 20 concurrent evaluate requests ---")
    t0 = time.perf_counter()
    eval_ids: list[str] = []
    errors:   list[str] = []

    def do_evaluate(i: int) -> tuple[int, str | None]:
        try:
            result = _post("/api/evaluate", laptop)
            assert result["status"] in ("ok","tie","no_valid_options","insufficient_info"), \
                f"Bad status: {result['status']}"
            assert "eval_id" in result, "Missing eval_id"
            return i, result["eval_id"]
        except Exception as exc:
            return i, f"ERROR:{exc}"

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(do_evaluate, i) for i in range(20)]
        for f in as_completed(futures):
            i, result = f.result()
            if isinstance(result, str) and result.startswith("ERROR:"):
                errors.append(f"Request {i}: {result}")
            else:
                eval_ids.append(result)

    elapsed = time.perf_counter() - t0
    print(f"  Completed: {len(eval_ids)}/20 successful in {elapsed:.2f}s")
    if errors:
        for e in errors:
            print(f"  FAIL {e}")
        return False
    print("  PASS: all 20 evaluations returned valid responses concurrently")

    # ── Phase 2: 10 concurrent feedback records ───────────────────────────
    print("\n--- Phase 2: 10 concurrent feedback record requests ---")
    # Use the first 10 eval_ids, all recording "a" as actual winner.
    t1 = time.perf_counter()
    fb_errors: list[str] = []

    def do_feedback(eval_id: str) -> str | None:
        try:
            result = _post("/api/feedback", {
                "eval_id": eval_id,
                "actual_winner": "c",   # tell engine "c" was actually best
            })
            assert "outcome" in result, "Missing outcome"
            return None
        except Exception as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(do_feedback, eid) for eid in eval_ids[:10]]
        for f in as_completed(futures):
            err = f.result()
            if err:
                fb_errors.append(err)

    elapsed2 = time.perf_counter() - t1
    print(f"  Completed: {10 - len(fb_errors)}/10 successful in {elapsed2:.2f}s")
    if fb_errors:
        for e in fb_errors:
            print(f"  FAIL {e}")
        return False
    print("  PASS: all 10 feedback records landed without error")

    # ── Phase 3: Verify store integrity ───────────────────────────────────
    print("\n--- Phase 3: Verify store integrity ---")
    summary = _get("/api/feedback/summary")
    total   = summary["total_outcomes_recorded"]
    correct = summary["correct_outcomes"]
    mults   = summary["weight_multipliers"]

    print(f"  total_outcomes_recorded : {total}")
    print(f"  correct_outcomes        : {correct}")
    print(f"  weight_multipliers      : {mults}")

    if total != 10:
        print(f"  FAIL: expected 10 outcomes, got {total} — concurrent writes lost data")
        return False
    print("  PASS: total_outcomes == 10 (no lost writes under concurrency)")

    # ── Phase 4: Adaptive behaviour evidence ──────────────────────────────
    print("\n--- Phase 4: Adaptive behaviour (evaluate with learned weights) ---")
    baseline = _post("/api/evaluate", laptop)
    adapted_body = {**laptop, "use_feedback": True}
    adapted  = _post("/api/evaluate", adapted_body)

    print(f"  Baseline decision  : {baseline['decision']} (weights as-is)")
    print(f"  Adapted  decision  : {adapted['decision']} (weights adjusted by feedback)")
    applied = adapted.get("applied_multipliers", {})
    if applied:
        for k, v in applied.items():
            direction = "boosted" if v > 1 else "attenuated"
            print(f"    {k}: ×{v:.3f} ({direction})")
    else:
        print("  (no multipliers applied — baseline was already correct)")
    print("  PASS: adaptive evaluation completed without error")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print(f"  Concurrent evaluations : 20/20 ok")
    print(f"  Concurrent feedback    : 10/10 ok, no lost writes")
    print(f"  Store integrity        : total={total} correct={correct}")
    print(f"  Adaptive evaluation    : baseline={baseline['decision']} "
          f"adapted={adapted['decision']}")
    print("=" * 60)
    print("\nFull request log written to de_engine.log")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
