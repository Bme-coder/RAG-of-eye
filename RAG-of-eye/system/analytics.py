"""
Statistical helpers for aggregating mined myopia progression records.
"""

from __future__ import annotations

import math
from typing import Iterable, Dict, Any, Optional

from config import (
    MIN_SAMPLE_SIZE,
    MAX_PROGRESSION_RATE,
    MIN_PROGRESSION_RATE,
    DEFAULT_SD_NATURAL,
    DEFAULT_SD_TREATMENT,
)


def _sanitize_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_weighted_statistics(
    raw_records: Iterable[Dict[str, Any]],
    *,
    is_treatment: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Compute weighted mean and pooled SD for a set of measurements.

    Args:
        raw_records: Iterable of dicts with keys mean, sd (optional), n, source_id.
        is_treatment: Whether to use treatment defaults for missing SDs.
    """
    cleaned = []
    default_sd = DEFAULT_SD_TREATMENT if is_treatment else DEFAULT_SD_NATURAL

    for record in raw_records:
        mean = _sanitize_float(record.get("mean"))
        sample_size = record.get("n") or record.get("sample_size")
        n = _sanitize_float(sample_size)
        if mean is None or n is None:
            continue
        n = int(n)
        if n < MIN_SAMPLE_SIZE:
            continue
        if mean < MAX_PROGRESSION_RATE or mean > MIN_PROGRESSION_RATE:
            continue

        sd_val = record.get("sd") or record.get("std") or record.get("stdev")
        sd = _sanitize_float(sd_val)
        if sd is None or sd <= 0:
            sd = default_sd

        cleaned.append(
            {
                "mean": mean,
                "sd": sd,
                "n": n,
                "source_id": record.get("source_id") or record.get("source") or record.get("paper_id"),
            }
        )

    if not cleaned:
        return None

    total_n = sum(item["n"] for item in cleaned)
    weighted_mean = sum(item["mean"] * item["n"] for item in cleaned) / total_n

    if total_n > 1:
        numerator = 0.0
        for item in cleaned:
            n = item["n"]
            numerator += (n - 1) * (item["sd"] ** 2) + n * ((item["mean"] - weighted_mean) ** 2)
        pooled_var = max(numerator / (total_n - 1), 1e-9)
    else:
        pooled_var = cleaned[0]["sd"] ** 2

    pooled_sd = math.sqrt(pooled_var)
    unique_sources = sorted({item["source_id"] for item in cleaned if item["source_id"]})

    return {
        "mean": round(weighted_mean, 4),
        "sd": round(pooled_sd, 4),
        "total_n": total_n,
        "records_used": len(cleaned),
        "source_ids": unique_sources,
    }
