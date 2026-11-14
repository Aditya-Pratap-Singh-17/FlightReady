"""Presentation-only diagnostics for immutable FlightReady v2 scorecards.

The frozen v2 method aggregates measured metric scores with a weighted geometric
mean.  Its historical ``measured_contribution`` field is an arithmetic
weighted-score diagnostic.  It must not be interpreted as an additive
contribution to the geometric score.  This module leaves the frozen scorecard
unchanged and supplies an explicit log-space explanation for reports.
"""
from __future__ import annotations

import math
from typing import Any


def geometric_influence(scorecard: dict[str, Any]) -> dict[str, Any]:
    """Explain a frozen scorecard's geometric aggregation without recalculating it.

    For positive measured scores, normalized log influences sum to
    ``ln(measured_quality_score / 100)``.  A measured zero makes the geometric
    score zero; logarithmic influences are then intentionally unavailable.
    """
    measured = [row for row in scorecard.get("metrics", []) if row.get("status") == "measured"]
    total_weight = sum(float(row.get("weight", 0.0)) for row in measured)
    zeros = [row.get("metric_id") for row in measured if float(row.get("score", 0.0)) == 0.0]
    rows: list[dict[str, Any]] = []
    for row in measured:
        weight = float(row["weight"])
        score = float(row["score"])
        normalized_weight = weight / total_weight if total_weight else None
        rows.append({
            "metric_id": row["metric_id"],
            "label": row.get("label", row["metric_id"]),
            "score": score,
            "normalized_weight": normalized_weight,
            "log_influence": None if zeros or normalized_weight is None else normalized_weight * math.log(score / 100),
        })
    if zeros:
        return {
            "aggregation": "weighted geometric mean",
            "status": "zero_score",
            "message": "At least one measured positive-weight metric is zero, so the geometric measured-quality score is zero. Log-space influence is undefined.",
            "zero_metric_ids": zeros,
            "metrics": rows,
        }
    log_sum = sum(float(row["log_influence"]) for row in rows) if rows else None
    reconstruction = 100 * math.exp(log_sum) if log_sum is not None else None
    return {
        "aggregation": "weighted geometric mean",
        "status": "explained" if rows else "no_measured_metrics",
        "message": "Log-space influences sum to ln(measured_quality_score / 100). They are explanatory diagnostics, not additive score points.",
        "log_influence_sum": log_sum,
        "reconstructed_measured_quality_score": reconstruction,
        "reported_measured_quality_score": scorecard.get("measured_quality_score"),
        "metrics": rows,
    }
