"""FlightReady v2 evidence-aware quality scorecards.

This module deliberately does not predict model accuracy.  It combines only
explicitly applicable measurements, keeps unavailable evidence in the
denominator, exposes every contribution, and derives the categorical decision
from gates rather than from the numeric summary.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

METHODOLOGY_VERSION = "flightready-v2.0"


class V2Metric(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    metric_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    dimension: str
    label: str
    weight: float = Field(gt=0, le=1000, strict=True)
    status: Literal["measured", "unavailable", "not_applicable", "error"]
    score: float | None = Field(default=None, ge=0, le=100)
    raw_value: Any = None
    normalization: str
    evidence_type: str
    reference_id: str | None = None
    limitation: str
    gate: Literal["block", "warning"] | None = None
    gate_passed: bool | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "measured" and self.score is None:
            raise ValueError("Measured metrics require a score")
        if self.status != "measured" and self.score is not None:
            raise ValueError("Unavailable/inapplicable/error metrics cannot be scored")
        if self.gate and self.status == "measured" and self.gate_passed is None:
            raise ValueError("Measured gates require an explicit outcome")
        return self


def _geometric(rows: list[V2Metric]) -> float | None:
    if not rows:
        return None
    if any(r.score == 0 for r in rows):
        return 0.0
    weight = sum(r.weight for r in rows)
    return 100 * math.exp(sum(r.weight * math.log(r.score / 100) for r in rows) / weight)


def build_scorecard(metrics: list[V2Metric], *, minimum_coverage: float = 0.8) -> dict[str, Any]:
    """Return an auditable scorecard; unknown evidence is never treated as pass.

    ``measured_quality_score`` summarizes measured criteria only.  The bounds
    show what the score could be if unavailable evidence were respectively 0
    or 100.  A primary score is published only when all applicable evidence is
    measured.  Gates, not the summary, determine readiness.
    """
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must lie in [0, 1]")
    ids = [m.metric_id for m in metrics]
    if len(ids) != len(set(ids)):
        raise ValueError("Metric IDs must be unique")
    applicable = [m for m in metrics if m.status != "not_applicable"]
    measured = [m for m in applicable if m.status == "measured"]
    total_weight = sum(m.weight for m in applicable)
    measured_weight = sum(m.weight for m in measured)
    coverage = measured_weight / total_weight if total_weight else 0.0
    measured_score = _geometric(measured)
    lower_rows = measured + [m.model_copy(update={"status": "measured", "score": 0.0}) for m in applicable if m.status != "measured"]
    upper_rows = measured + [m.model_copy(update={"status": "measured", "score": 100.0}) for m in applicable if m.status != "measured"]
    blocking_failures = [m.metric_id for m in measured if m.gate == "block" and m.gate_passed is False]
    blocking_unknown = [m.metric_id for m in applicable if m.gate == "block" and m.status != "measured"]
    errors = [m.metric_id for m in applicable if m.status == "error"]
    warnings = [m.metric_id for m in measured if m.gate == "warning" and m.gate_passed is False]
    if errors:
        decision = "ASSESSMENT_ERROR"
    elif blocking_failures:
        decision = "NOT_READY"
    elif blocking_unknown or coverage < minimum_coverage:
        decision = "INSUFFICIENT_EVIDENCE"
    elif warnings:
        decision = "CONDITIONALLY_READY"
    else:
        decision = "READY"
    dimensions = []
    for dimension in sorted({m.dimension for m in metrics}):
        members = [m for m in metrics if m.dimension == dimension and m.status != "not_applicable"]
        observed = [m for m in members if m.status == "measured"]
        denominator = sum(m.weight for m in members)
        dimensions.append({
            "dimension": dimension,
            "score": _geometric(observed) if observed and len(observed) == len(members) else None,
            "measured_score": _geometric(observed),
            "evidence_coverage": sum(m.weight for m in observed) / denominator if denominator else 0.0,
            "applicable_metrics": len(members),
            "measured_metrics": len(observed),
        })
    rows = []
    for m in metrics:
        row = m.model_dump(mode="json")
        row["normalized_weight"] = m.weight / total_weight if m in applicable and total_weight else 0.0
        row["measured_contribution"] = (
            m.weight / measured_weight * m.score if m in measured and measured_weight else None
        )
        rows.append(row)
    result = {
        "methodology_version": METHODOLOGY_VERSION,
        "readiness_decision": decision,
        "primary_quality_score": measured_score if coverage == 1 else None,
        "measured_quality_score": measured_score,
        "score_bounds": [_geometric(lower_rows), _geometric(upper_rows)] if applicable else [None, None],
        "evidence_coverage": coverage,
        "minimum_decision_coverage": minimum_coverage,
        "applicable_metric_count": len(applicable),
        "measured_metric_count": len(measured),
        "unavailable_metric_ids": [m.metric_id for m in applicable if m.status == "unavailable"],
        "blocking_failures": blocking_failures,
        "blocking_unknown": blocking_unknown,
        "warnings": warnings,
        "dimensions": dimensions,
        "metrics": rows,
        "interpretation": "Quality criteria summary, not predicted model accuracy. Missing evidence is retained and is not a pass.",
    }
    result["calculation_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return result


def fraction_metric(metric_id: str, dimension: str, label: str, bad_fraction: float | None,
                    *, weight: float = 1.0, evidence_type: str = "observed",
                    reference_id: str | None = None, limitation: str,
                    gate: Literal["block", "warning"] | None = None,
                    maximum: float | None = None) -> V2Metric:
    """Create a lower-is-better fraction metric with strict validation."""
    if bad_fraction is None:
        return V2Metric(metric_id=metric_id, dimension=dimension, label=label, weight=weight,
                        status="unavailable", normalization="100 * (1 - fraction)",
                        evidence_type=evidence_type, reference_id=reference_id, limitation=limitation,
                        gate=gate)
    if isinstance(bad_fraction, bool) or not isinstance(bad_fraction, (int, float)) or not math.isfinite(bad_fraction) or not 0 <= bad_fraction <= 1:
        raise ValueError("Fraction evidence must be a finite number in [0, 1]")
    return V2Metric(metric_id=metric_id, dimension=dimension, label=label, weight=weight,
                    status="measured", score=100 * (1 - bad_fraction), raw_value=bad_fraction,
                    normalization="100 * (1 - fraction)", evidence_type=evidence_type,
                    reference_id=reference_id, limitation=limitation, gate=gate,
                    gate_passed=None if gate is None else bad_fraction <= (maximum if maximum is not None else 0.0))
