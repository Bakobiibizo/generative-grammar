"""Simple reproducible grammar-based synthetic data generators."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _coerce_scalar(value: object, caster: type, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, caster):
        return value
    try:
        if caster is bool:
            if isinstance(value, bool):
                return value
            return str(value).lower() in {"1", "true", "yes", "on"}
        if caster in (int, float):
            return caster(value)
        if caster is str:
            return str(value)
    except Exception:
        return default
    return default


def _parse_config(config: dict[str, str] | None) -> dict[str, Any]:
    cfg = config or {}
    get = cfg.get

    return {
        "seed_list": [
            int(v)
            for v in str(get("experiment.seed_list", "42,123,456,789,1337")).split(",")
            if str(v).strip()
        ],
        "kind": _coerce_scalar(get("grammar.kind"), str, "balanced_parentheses"),
        "sample_count": _coerce_scalar(get("grammar.sample_count"), int, 120),
        "max_depth": _coerce_scalar(get("grammar.max_depth"), int, 6),
        "max_length": _coerce_scalar(get("grammar.max_length"), int, 40),
    }


def _balanced_parentheses(rng: random.Random, max_depth: int, max_length: int) -> str:
    target = rng.randint(2, max(2, min(2 * max_depth, max_length)))
    if target % 2 == 1:
        target += 1
    current = ""
    depth = 0
    remaining = target
    for _ in range(max_length):
        if remaining <= 0:
            break
        if depth == 0:
            action = "("
        elif depth * 2 >= remaining:
            action = ")"
        elif depth >= max_depth:
            action = ")"
        else:
            # Slight bias to keep sequences valid and varied.
            action = "(" if rng.random() < 0.55 else ")"

        current += action
        remaining -= 1
        depth += 1 if action == "(" else -1
        if depth == 0 and len(current) >= target:
            break
    return current


def _arith_expr(rng: random.Random, max_depth: int, max_length: int) -> str:
    ops = ["+", "-", "*", "/"]
    terms = [str(rng.randint(0, 9)) for _ in range(max(2, max_depth + 2))]
    out = terms[0]
    for i in range(1, len(terms)):
        out += rng.choice(ops) + terms[i]
        if len(out) >= max_length:
            break
    # Optionally add a short nested form.
    if rng.random() < 0.35 and len(out) + 2 <= max_length:
        out = f"({out})"
    return out


def _ab_sequence(rng: random.Random, max_length: int) -> str:
    length = rng.randint(3, max(3, max_length))
    return "".join(rng.choice(["a", "b", "c"]) for _ in range(length))


def _generate_one(kind: str, rng: random.Random, max_depth: int, max_length: int) -> str:
    if kind == "balanced_parentheses":
        return _balanced_parentheses(rng, max_depth=max_depth, max_length=max_length)
    if kind == "arithmetic":
        return _arith_expr(rng, max_depth=max_depth, max_length=max_length)
    if kind == "alphabetic":
        return _ab_sequence(rng, max_length=max_length)
    return _balanced_parentheses(rng, max_depth=max_depth, max_length=max_length)


def _is_valid_balanced(value: str) -> bool:
    depth = 0
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _is_valid(value: str, kind: str) -> bool:
    if kind == "balanced_parentheses":
        return value and _is_valid_balanced(value)
    if kind == "arithmetic":
        # Quick heuristic check: at least one digit and allowed symbols.
        return bool(value) and all(ch.isdigit() or ch in "+-*/()" for ch in value)
    if kind == "alphabetic":
        return bool(value) and set(value).issubset({"a", "b", "c"})
    return bool(value)


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs)))


def run(seed: int, output_dir: Path, config: dict[str, str] | None = None) -> dict[str, Any]:
    parsed = _parse_config(config)
    kind = parsed["kind"]
    sample_count = parsed["sample_count"]
    max_depth = parsed["max_depth"]
    max_length = parsed["max_length"]

    rng = random.Random(seed)
    samples = [
        {
            "seed": seed,
            "idx": i,
            "grammar": kind,
            "value": _generate_one(kind, rng, max_depth=max_depth, max_length=max_length),
        }
        for i in range(sample_count)
    ]

    values = [row["value"] for row in samples]
    valid = [row for row in samples if _is_valid(row["value"], kind)]

    table = pa.Table.from_pydict(
        {
            "seed": pa.array([r["seed"] for r in samples], type=pa.int64()),
            "idx": pa.array([r["idx"] for r in samples], type=pa.int64()),
            "grammar": pa.array([r["grammar"] for r in samples], type=pa.string()),
            "value": pa.array([r["value"] for r in samples], type=pa.string()),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_dir / f"grammar_samples_{kind}_{seed}.parquet")

    valid_rate = len(valid) / sample_count if sample_count else 0.0
    mean_len = float(np.mean([len(v) for v in values])) if values else 0.0
    entropy = _entropy(values)

    # Additional lightweight metrics.
    max_unique = len(set(values))
    collision = 1.0 - (max_unique / max(1, sample_count))

    return {
        "seed": seed,
        "h_generate_valid_rate": valid_rate,
        "h_generate_mean_length": mean_len,
        "h_generate_entropy": entropy,
        "h_generate_collision_rate": collision,
        "sample_count": sample_count,
    }
