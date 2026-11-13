import pytest

from metrics.readiness_v2 import V2Metric, build_scorecard, fraction_metric


def metric(metric_id="a", score=100.0, **kwargs):
    return V2Metric(metric_id=metric_id, dimension=kwargs.pop("dimension", "integrity"),
                    label=metric_id, weight=kwargs.pop("weight", 1.0), status="measured",
                    score=score, raw_value=score, normalization="test",
                    evidence_type="unit_test", limitation="Synthetic regression fixture", **kwargs)


def test_missing_evidence_is_not_passed_or_renormalized_as_complete():
    unknown = V2Metric(metric_id="unknown", dimension="integrity", label="unknown", weight=1.0,
                       status="unavailable", normalization="test", evidence_type="unit_test",
                       limitation="No source")
    result = build_scorecard([metric(score=80), unknown])
    assert result["primary_quality_score"] is None
    assert result["measured_quality_score"] == pytest.approx(80)
    assert result["evidence_coverage"] == pytest.approx(.5)
    assert result["score_bounds"] == pytest.approx([0, 89.4427191])
    assert result["readiness_decision"] == "INSUFFICIENT_EVIDENCE"


def test_confirmed_failure_wins_over_missing_evidence():
    failed = metric(gate="block", gate_passed=False)
    unknown = V2Metric(metric_id="unknown", dimension="coverage", label="unknown", weight=1.0,
                       status="unavailable", normalization="test", evidence_type="unit_test",
                       limitation="No manifest", gate="block")
    assert build_scorecard([failed, unknown])["readiness_decision"] == "NOT_READY"


def test_geometric_score_preserves_real_zero_and_contributions():
    result = build_scorecard([metric("good"), metric("bad", 0.0)])
    assert result["primary_quality_score"] == 0
    assert result["metrics"][0]["measured_contribution"] == 50
    assert result["metrics"][1]["measured_contribution"] == 0


def test_fraction_metric_direction_and_gate():
    row = fraction_metric("missing", "integrity", "Missing", .2, limitation="Observed cells",
                          gate="block", maximum=.1)
    assert row.score == 80
    assert row.gate_passed is False


def test_rejects_duplicate_ids_and_invalid_fractions():
    with pytest.raises(ValueError, match="unique"):
        build_scorecard([metric(), metric()])
    with pytest.raises(ValueError, match="finite"):
        fraction_metric("x", "d", "x", float("nan"), limitation="x")


def test_certificate_exposes_versioned_v2_profile():
    from backend.readiness_engine import assess
    from backend.context import DatasetContext, Goal
    from backend.ingestion import load_dataset
    dataset = load_dataset("synthetic.csv", b"x,label\n1,car\n2,pedestrian\n3,car\n")
    context = DatasetContext(original_text="Synthetic test goal",
                             goal=Goal(task_type="classification", prediction_target="label", required_inputs=["x"]),
                             confirmed=True)
    certificate = assess(dataset, context)
    assert certificate.quality_profile_v2["methodology_version"] == "flightready-v2.0"
    assert certificate.quality_profile_v2["interpretation"].startswith("Quality criteria")
