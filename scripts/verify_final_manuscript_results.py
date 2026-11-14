"""Independently verify FlightReady manuscript numbers from saved artifacts.

This script is read-only. It does not regenerate or modify experiment records.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"


def _close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None
    sem = stats.sem(values)
    if not np.isfinite(sem):
        return float(values[0]), float(values[0])
    half = stats.t.ppf(0.975, len(values) - 1) * sem
    return float(values.mean() - half), float(values.mean() + half)


def _planned_tags(config):
    tr = config["training"]
    plan = config["executed_resource_plan"]
    rows = []
    for condition in tr["primary_conditions"]:
        seeds = plan["repeat_seeds"] if condition in plan["repeat_seed_conditions"] else [plan["other_primary_seed"]]
        rows.extend(("yolo11n", condition, seed, "fixed_epochs") for seed in seeds)
    rows.extend(("yolo11n", condition, plan["severity_seed"], "fixed_epochs")
                for condition in tr["severity_conditions_seed_17"])
    rows.extend(("yolo11n", tr["fixed_update_condition"], seed, "fixed_updates")
                for seed in plan["fixed_update_seeds"])
    rows.extend(("yolov8n", condition, seed, "fixed_epochs")
                for condition in tr["secondary_conditions"] for seed in plan["secondary_seeds"])
    return [f"{model}__{condition}__seed{seed}__{schedule}__batch{tr['batch']}__workers{tr['workers']}"
            for model, condition, seed, schedule in rows]


def _load_runs(config):
    runs = []
    for path in (EXP / "model_results").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("pilot") or row.get("batch") != config["training"]["batch"] or row.get("workers") != config["training"]["workers"]:
            continue
        metrics = row["test_metrics"]
        runs.append({
            "tag": row["tag"], "model": row["model"], "condition": row["condition"],
            "seed": row["seed"], "schedule": row["schedule"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "map50": metrics["map50"], "map50_95": metrics["map50_95"],
            "elapsed_seconds": row["elapsed_seconds"], "result_file": str(path),
        })
    runs.sort(key=lambda r: (r["model"], r["condition"], r["schedule"], r["seed"], r["tag"]))
    assert len({r["tag"] for r in runs}) == len(runs), "duplicate final-protocol tags"
    return runs


def _summaries(runs):
    clean = {(r["model"], r["seed"]): r for r in runs
             if r["condition"] == "clean" and r["schedule"] == "fixed_epochs"}
    for r in runs:
        match = clean.get((r["model"], r["seed"]))
        r["paired_difference"] = r["map50_95"] - match["map50_95"] if match else None
        r["paired_relative_percent"] = 100 * (r["map50_95"] / match["map50_95"] - 1) if match else None
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["condition"], r["schedule"])].append(r)
    out = {}
    for key, members in groups.items():
        rec = {"n": len(members), "seeds": sorted(r["seed"] for r in members)}
        for metric in ("precision", "recall", "map50", "map50_95", "elapsed_seconds"):
            values = [r[metric] for r in members]
            low, high = _interval(values)
            rec[f"{metric}_mean"] = float(np.mean(values))
            rec[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            rec[f"{metric}_ci95_low"] = low
            rec[f"{metric}_ci95_high"] = high
        paired = [r["paired_difference"] for r in members if r["paired_difference"] is not None]
        paired_rel = [r["paired_relative_percent"] for r in members if r["paired_relative_percent"] is not None]
        low, high = _interval(paired)
        rec.update({
            "paired_clean_n": len(paired),
            "paired_map50_95_difference_mean": float(np.mean(paired)) if paired else None,
            "paired_map50_95_difference_sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else None,
            "paired_map50_95_difference_ci95_low": low,
            "paired_map50_95_difference_ci95_high": high,
            "paired_map50_95_relative_change_percent_mean": float(np.mean(paired_rel)) if paired_rel else None,
        })
        out[key] = rec
    return out


def _geometric(metrics):
    if not metrics:
        return None
    if any(float(m["score"]) == 0 for m in metrics):
        return 0.0
    denominator = sum(float(m["weight"]) for m in metrics)
    numerator = sum(float(m["weight"]) * math.log(float(m["score"]) / 100) for m in metrics)
    return 100 * math.exp(numerator / denominator)


def _verify_scorecard(card):
    applicable = [m for m in card["metrics"] if m["status"] != "not_applicable"]
    measured = [m for m in applicable if m["status"] == "measured"]
    total_weight = sum(float(m["weight"]) for m in applicable)
    measured_weight = sum(float(m["weight"]) for m in measured)
    coverage = measured_weight / total_weight if total_weight else 0.0
    measured_score = _geometric(measured)
    assert _close(card["evidence_coverage"], coverage)
    assert _close(card["measured_quality_score"], measured_score)
    expected_primary = measured_score if coverage == 1 else None
    assert _close(card["primary_quality_score"], expected_primary)
    failures = [m["metric_id"] for m in measured if m.get("gate") == "block" and m.get("gate_passed") is False]
    unknown = [m["metric_id"] for m in applicable if m.get("gate") == "block" and m["status"] != "measured"]
    errors = [m["metric_id"] for m in applicable if m["status"] == "error"]
    warnings = [m["metric_id"] for m in measured if m.get("gate") == "warning" and m.get("gate_passed") is False]
    decision = ("ASSESSMENT_ERROR" if errors else "NOT_READY" if failures else
                "INSUFFICIENT_EVIDENCE" if unknown or coverage < card["minimum_decision_coverage"] else
                "CONDITIONALLY_READY" if warnings else "READY")
    assert card["readiness_decision"] == decision
    return {"coverage": coverage, "measured_score": measured_score,
            "primary_score": expected_primary, "decision": decision, "warnings": warnings}


def verify():
    config = json.loads((EXP / "experiment_config.json").read_text(encoding="utf-8"))
    quality = json.loads((EXP / "quality_results" / "flightready_assessments.json").read_text(encoding="utf-8"))["variants"]
    saved_stats = json.loads((EXP / "analysis" / "statistical_analysis.json").read_text(encoding="utf-8"))
    runs = _load_runs(config)
    summaries = _summaries(runs)
    planned = _planned_tags(config)
    observed = {r["tag"] for r in runs}
    completed = sorted(set(planned) & observed)
    missing = sorted(set(planned) - observed)
    supplemental = sorted(observed - set(planned))
    assert len(planned) == 19 and len(completed) == 19 and not missing
    assert len(runs) == 21 and supplemental == [
        "yolo11n__gaussian_25__seed42__fixed_epochs__batch32__workers2",
        "yolo11n__gaussian_25__seed93__fixed_epochs__batch32__workers2",
    ]

    with (EXP / "analysis" / "condition_summary.csv").open(encoding="utf-8") as f:
        stored_rows = list(csv.DictReader(f))
    checked_fields = [
        "n", "precision_mean", "precision_sd", "precision_ci95_low", "precision_ci95_high",
        "recall_mean", "recall_sd", "recall_ci95_low", "recall_ci95_high",
        "map50_mean", "map50_sd", "map50_ci95_low", "map50_ci95_high",
        "map50_95_mean", "map50_95_sd", "map50_95_ci95_low", "map50_95_ci95_high",
        "paired_clean_n", "paired_map50_95_difference_mean", "paired_map50_95_difference_sd",
        "paired_map50_95_difference_ci95_low", "paired_map50_95_difference_ci95_high",
        "paired_map50_95_relative_change_percent_mean",
    ]
    for row in stored_rows:
        key = (row["model"], row["condition"], row["schedule"])
        rec = summaries[key]
        for field in checked_fields:
            stored = None if row[field] == "" else float(row[field])
            recalculated = rec[field]
            assert _close(stored, recalculated), (key, field, stored, recalculated)

    primary = config["training"]["primary_conditions"]
    scores = [quality[c]["scorecards"]["full_oracle_assisted"]["primary_quality_score"] for c in primary]
    performance = [summaries[("yolo11n", c, "fixed_epochs")]["map50_95_mean"] for c in primary]
    pearson = stats.pearsonr(scores, performance)
    spearman = stats.spearmanr(scores, performance)
    keep = [i for i, c in enumerate(primary) if c != "repaired"]
    pearson7 = stats.pearsonr([scores[i] for i in keep], [performance[i] for i in keep])
    spearman7 = stats.spearmanr([scores[i] for i in keep], [performance[i] for i in keep])
    corr = {
        "pearson": {"n": 8, "coefficient": float(pearson.statistic), "p_value": float(pearson.pvalue)},
        "spearman": {"n": 8, "coefficient": float(spearman.statistic), "p_value": float(spearman.pvalue)},
        "without_repaired_pearson": {"n": 7, "coefficient": float(pearson7.statistic), "p_value": float(pearson7.pvalue)},
        "without_repaired_spearman": {"n": 7, "coefficient": float(spearman7.statistic), "p_value": float(spearman7.pvalue)},
    }
    saved_corr = saved_stats["correlations"]
    for ours, saved in [
        (corr["pearson"], saved_corr["oracle_score_vs_map50_95"]["pearson"]),
        (corr["spearman"], saved_corr["oracle_score_vs_map50_95"]["spearman"]),
        (corr["without_repaired_pearson"], saved_corr["sensitivity_excluding_repaired_duplicate_control"]["pearson"]),
        (corr["without_repaired_spearman"], saved_corr["sensitivity_excluding_repaired_duplicate_control"]["spearman"]),
    ]:
        assert ours["n"] == saved["n"] and _close(ours["coefficient"], saved["coefficient"]) and _close(ours["p_value"], saved["p_value"])

    scorecards = {}
    for variant, record in quality.items():
        scorecards[variant] = {mode: _verify_scorecard(card) for mode, card in record["scorecards"].items()}
    defects = [v for v in config["variants"] if v not in {"clean", "repaired"}]
    detection = {}
    for mode in ("reference_free", "deterministic_10_percent_annotation_audit", "full_oracle_assisted"):
        clean = quality["clean"]["scorecards"][mode]
        clean_score = clean["primary_quality_score"] if clean["primary_quality_score"] is not None else clean["measured_quality_score"]
        found = []
        for variant in defects:
            card = quality[variant]["scorecards"][mode]
            score = card["primary_quality_score"] if card["primary_quality_score"] is not None else card["measured_quality_score"]
            if score < clean_score - 0.1 or set(card["warnings"]) != set(clean["warnings"]):
                found.append(variant)
        detection[mode] = {"detected": found, "count": len(found), "total": len(defects), "sensitivity": len(found) / len(defects)}
        saved = saved_stats["detection_analysis"][mode]
        assert found == saved["detected"] and _close(detection[mode]["sensitivity"], saved["sensitivity"])

    baseline = []
    for variant in defects:
        flags = quality[variant]["conventional_baselines"]
        baseline.append({"variant": variant, "detected": any(flags.values()), "positive_checks": [k for k, v in flags.items() if v]})
    assert baseline == saved_stats["conventional_baseline_detection"]

    clean = summaries[("yolo11n", "clean", "fixed_epochs")]
    combined = summaries[("yolo11n", "combined_40", "fixed_epochs")]
    gaussian = summaries[("yolo11n", "gaussian_25", "fixed_epochs")]
    return {
        "planned_runs": len(planned), "completed_planned_runs": len(completed),
        "observed_runs": len(runs), "supplemental_tags": supplemental,
        "all_saved_summary_values_match": True,
        "correlations": corr,
        "detection": detection,
        "conventional_baselines": {"detected": sum(r["detected"] for r in baseline), "total": len(baseline), "details": baseline},
        "reference_free": {"all_primary_scores_withheld": all(scorecards[v]["reference_free"]["primary_score"] is None for v in quality),
                           "coverage_values": sorted({round(scorecards[v]["reference_free"]["coverage"], 12) for v in quality})},
        "key_results": {
            "clean": clean, "combined_40": combined, "gaussian_25": gaussian,
            "coverage_40_fixed_updates": summaries[("yolo11n", "coverage_40", "fixed_updates")],
            "yolov8_clean": summaries[("yolov8n", "clean", "fixed_epochs")],
            "yolov8_combined_40": summaries[("yolov8n", "combined_40", "fixed_epochs")],
        },
        "scorecards": scorecards,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
