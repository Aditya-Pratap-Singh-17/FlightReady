import pytest

from metrics.readiness_v2 import V2Metric, build_scorecard
from metrics.readiness_v2_reporting import geometric_influence


def metric(metric_id, score, weight):
    return V2Metric(metric_id=metric_id, dimension="integrity", label=metric_id,
                    weight=weight, status="measured", score=score,
                    normalization="test", evidence_type="unit_test", limitation="fixture")


def test_log_influences_reconstruct_the_frozen_geometric_score():
    scorecard = build_scorecard([metric("a", 80.0, 1.0), metric("b", 50.0, 3.0)])
    explanation = geometric_influence(scorecard)
    assert explanation["status"] == "explained"
    assert explanation["reconstructed_measured_quality_score"] == pytest.approx(scorecard["measured_quality_score"])
    assert sum(row["log_influence"] for row in explanation["metrics"]) == pytest.approx(explanation["log_influence_sum"])
    assert sum(row["measured_contribution"] for row in scorecard["metrics"]) != pytest.approx(scorecard["measured_quality_score"])


def test_zero_metric_is_explained_without_false_additive_contribution():
    scorecard = build_scorecard([metric("good", 100.0, 1.0), metric("bad", 0.0, 1.0)])
    explanation = geometric_influence(scorecard)
    assert explanation["status"] == "zero_score"
    assert explanation["zero_metric_ids"] == ["bad"]
    assert all(row["log_influence"] is None for row in explanation["metrics"])
