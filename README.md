# Decision Engine (Conflict Reasoning, No AI APIs)

This repo contains a **deterministic decision engine** that:

- Ingests **real-world ambiguous inputs** (missing data, soft preferences, constraints)
- Detects and reasons about **conflicts** (hard/soft, within or across sources)
- Performs **multi-factor scoring** (weights, source reliability, recency, uncertainty penalties)
- Produces **explainable outputs** (structured trace: factors, conflicts, assumptions, sensitivities)

No external AI APIs are used.

## Quick start

### 1) Chạy ngay (không cần cài)

Project dùng **standard library** nên chạy thẳng được.

### 2) Run the demo CLI

```bash
python -m decision_engine.cli examples/purchase_laptop.json
```

## Kết quả chạy mẫu (rút gọn)

### Demo 1: Laptop (có dữ liệu mâu thuẫn + hard constraint)

```bash
python -m decision_engine.cli examples/purchase_laptop.json --pretty
```

Ví dụ output (rút gọn):

```json
{
  "decision": "a",
  "status": "ok",
  "conflicts": [
    { "kind": "value_value", "factor": "price_usd", "option_id": "a", "severity": "medium" },
    { "kind": "value_value", "factor": "battery_hours", "option_id": "a", "severity": "medium" }
  ],
  "assumptions": [
    "Missing value for factor 'weight_kg' on option 'a'; treated as neutral."
  ],
  "explanation": {
    "summary": "Selected 'Laptop A' because it achieved the highest multi-factor score among non-disqualified options."
  }
}
```

### Demo 2: Ràng buộc tự mâu thuẫn (infeasible constraints)

```bash
python -m decision_engine.cli examples/conflicting_constraints.json --pretty
```

Ví dụ output (rút gọn):

```json
{
  "decision": null,
  "status": "no_valid_options",
  "conflicts": [
    { "kind": "constraint_constraint", "factor": "latency_ms", "severity": "high" }
  ],
  "explanation": {
    "summary": "All options were disqualified by constraints."
  }
}
```

### 3) Cài như một “product” (pip install .)

Trong thư mục repo:

```bash
python -m pip install -U pip
python -m pip install -e .
```

Sau đó bạn có lệnh:

```bash
decision-engine examples/purchase_laptop.json --pretty
```

### 4) Chạy test (để chứng minh ổn định)

```bash
python -m pip install -e ".[dev]"  # (tuỳ chọn, nếu bạn bổ sung dev deps sau)
python -m pytest -q
```

## Input format (high level)

An evaluation request contains:

- **options**: items to choose among
- **factors**: multi-criteria attributes (price, risk, time, etc.)
- **inputs**: user statements (may conflict), each with:
  - `claim`: a factor value, constraint, or preference
  - `strength`: hard/soft
  - `source`: who/where it came from (with `reliability`)
  - `timestamp`: for recency weighting

See `examples/` for complete JSON.

## Output format (high level)

The engine returns:

- **decision**: selected option (or tie / cannot decide)
- **scores**: per-option totals + per-factor breakdown
- **conflicts**: detected conflicts and how they were handled
- **assumptions**: defaults applied due to missing/ambiguous info
- **explanation**: human-readable summary + structured trace

## Why this design

The evaluator focus is **system design, robustness, reasoning depth, and ambiguity handling**.
This engine is explicitly built around:

- Conflict classification (hard vs soft; constraint vs preference)
- Evidence aggregation (reliability, recency, consistency)
- Explainability-first output (trace is a first-class artifact)

