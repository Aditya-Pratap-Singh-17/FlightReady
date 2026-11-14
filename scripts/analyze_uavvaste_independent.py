"""Aggregate completed model runs and compute preregistered descriptive statistics."""
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
CONFIG = json.loads((EXP / "experiment_config.json").read_text(encoding="utf-8"))
QUALITY = json.loads((EXP / "quality_results" / "flightready_assessments.json").read_text(encoding="utf-8"))["variants"]
OUT = EXP / "analysis"


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None
    sem = stats.sem(values)
    if not np.isfinite(sem):
        return float(values[0]), float(values[0])
    half = stats.t.ppf(0.975, len(values) - 1) * sem
    return float(values.mean() - half), float(values.mean() + half)


def correlation(x, y, kind):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "coefficient": None, "p_value": None, "reason": "insufficient observations or zero variance"}
    result = stats.pearsonr(x, y) if kind == "pearson" else stats.spearmanr(x, y)
    return {"n": len(x), "coefficient": float(result.statistic), "p_value": float(result.pvalue)}


def planned_tags():
    """Return the declared final-protocol tags without importing the trainer."""
    training = CONFIG["training"]
    reduced = CONFIG.get("executed_resource_plan")
    plans = []
    for condition in training["primary_conditions"]:
        seeds = (reduced["repeat_seeds"] if reduced and condition in reduced["repeat_seed_conditions"]
                 else [reduced["other_primary_seed"]] if reduced else training["primary_seeds"])
        plans.extend(("yolo11n", condition, seed, training["epochs_fixed"], "fixed_epochs") for seed in seeds)
    plans.extend(("yolo11n", condition, 17, training["epochs_fixed"], "fixed_epochs")
                 for condition in training["severity_conditions_seed_17"])
    update_seeds = reduced["fixed_update_seeds"] if reduced else training["fixed_update_seeds"]
    plans.extend(("yolo11n", training["fixed_update_condition"], seed, training["fixed_update_epochs"], "fixed_updates")
                 for seed in update_seeds)
    secondary_seeds = reduced["secondary_seeds"] if reduced else training["secondary_seeds"]
    plans.extend(("yolov8n", condition, seed, training["epochs_fixed"], "fixed_epochs")
                 for condition in training["secondary_conditions"] for seed in secondary_seeds)
    batch, workers = training["batch"], training["workers"]
    return [f"{model}__{condition}__seed{seed}__{schedule}__batch{batch}__workers{workers}"
            for model, condition, seed, _, schedule in plans]


def main():
    batch = CONFIG["training"]["batch"]
    workers = CONFIG["training"]["workers"]
    runs = []
    for path in (EXP / "model_results").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("pilot") or row.get("batch") != batch or row.get("workers") != workers:
            continue
        flat = {k: row[k] for k in ("tag", "model", "condition", "seed", "schedule", "planned_epochs", "batch", "workers", "elapsed_seconds")}
        flat.update(row["test_metrics"])
        flat["result_file"] = path.relative_to(ROOT).as_posix()
        runs.append(flat)
    runs.sort(key=lambda r: (r["model"], r["condition"], r["schedule"], r["seed"], r["tag"]))
    clean_by_model_seed = {(r["model"], r["seed"]): r for r in runs
                           if r["condition"] == "clean" and r["schedule"] == "fixed_epochs"}
    for row in runs:
        clean = clean_by_model_seed.get((row["model"], row["seed"]))
        row["map50_95_absolute_change"] = row["map50_95"] - clean["map50_95"] if clean else None
        row["map50_95_relative_change_percent"] = (
            100 * (row["map50_95"] / clean["map50_95"] - 1) if clean and clean["map50_95"] else None
        )
    groups = defaultdict(list)
    for row in runs:
        groups[(row["model"], row["condition"], row["schedule"])].append(row)
    summary = []
    for (model, condition, schedule), members in sorted(groups.items()):
        record = {"model": model, "condition": condition, "schedule": schedule, "n": len(members), "seeds": sorted(m["seed"] for m in members)}
        for metric in ("precision", "recall", "map50", "map50_95", "elapsed_seconds"):
            values = [m[metric] for m in members]
            low, high = interval(values)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            record[f"{metric}_ci95_low"] = low
            record[f"{metric}_ci95_high"] = high
        paired = [m["map50_95_absolute_change"] for m in members if m["map50_95_absolute_change"] is not None]
        paired_relative = [m["map50_95_relative_change_percent"] for m in members if m["map50_95_relative_change_percent"] is not None]
        low, high = interval(paired)
        record["paired_clean_n"] = len(paired)
        record["paired_map50_95_difference_mean"] = float(np.mean(paired)) if paired else None
        record["paired_map50_95_difference_sd"] = float(np.std(paired, ddof=1)) if len(paired) > 1 else None
        record["paired_map50_95_difference_ci95_low"] = low
        record["paired_map50_95_difference_ci95_high"] = high
        record["paired_map50_95_relative_change_percent_mean"] = float(np.mean(paired_relative)) if paired_relative else None
        summary.append(record)

    primary = CONFIG["training"]["primary_conditions"]
    primary_means = {r["condition"]: r for r in summary if r["model"] == "yolo11n" and r["schedule"] == "fixed_epochs" and r["condition"] in primary}
    ordered = [c for c in primary if c in primary_means]
    scores = [QUALITY[c]["scorecards"]["full_oracle_assisted"]["primary_quality_score"] for c in ordered]
    performance = [primary_means[c]["map50_95_mean"] for c in ordered]
    no_repair = [i for i, c in enumerate(ordered) if c != "repaired"]
    correlation_results = {
        "unit": "dataset-condition mean; model seeds are repeated measurements, not independent datasets",
        "primary_conditions": ordered,
        "oracle_score_vs_map50_95": {
            "pearson": correlation(scores, performance, "pearson"),
            "spearman": correlation(scores, performance, "spearman"),
        },
        "sensitivity_excluding_repaired_duplicate_control": {
            "pearson": correlation([scores[i] for i in no_repair], [performance[i] for i in no_repair], "pearson"),
            "spearman": correlation([scores[i] for i in no_repair], [performance[i] for i in no_repair], "spearman"),
        },
        "reference_free_score_correlation": {"coefficient": None, "reason": "primary scores withheld for all conditions because evidence coverage is below 0.8"},
    }

    defect_variants = [v for v in CONFIG["variants"] if v not in {"clean", "repaired"}]
    modes = ["reference_free", "full_oracle_assisted", "deterministic_10_percent_annotation_audit"]
    detection = {}
    for mode in modes:
        clean_card = QUALITY["clean"]["scorecards"][mode]
        clean_score = clean_card["primary_quality_score"] if clean_card["primary_quality_score"] is not None else clean_card["measured_quality_score"]
        detected = []
        for variant in defect_variants:
            card = QUALITY[variant]["scorecards"][mode]
            score = card["primary_quality_score"] if card["primary_quality_score"] is not None else card["measured_quality_score"]
            if score < clean_score - 0.1 or set(card["warnings"]) != set(clean_card["warnings"]):
                detected.append(variant)
        repaired = QUALITY["repaired"]["scorecards"][mode]
        repaired_score = repaired["primary_quality_score"] if repaired["primary_quality_score"] is not None else repaired["measured_quality_score"]
        detection[mode] = {"defects_total": len(defect_variants), "detected": detected, "sensitivity": len(detected) / len(defect_variants),
                           "clean_false_positive": False, "repaired_false_positive": repaired_score < clean_score - 0.1 or set(repaired["warnings"]) != set(clean_card["warnings"]),
                           "definition": "score decrease >0.1 points or changed warning set relative to clean"}
    conventional = []
    for variant in defect_variants:
        flags = QUALITY[variant]["conventional_baselines"]
        conventional.append({"variant": variant, "detected": any(flags.values()), "positive_checks": [k for k, v in flags.items() if v]})

    expected_tags = planned_tags()
    observed_tags = {row["tag"] for row in runs}
    completed_tags = sorted(observed_tags.intersection(expected_tags))
    missing_tags = sorted(set(expected_tags) - observed_tags)
    result = {
        "protocol_version": CONFIG["protocol_version"], "final_protocol": {"batch": batch, "workers": workers},
        "planned_model_runs": len(expected_tags),
        "completed_final_protocol_runs": len(completed_tags),
        "observed_final_protocol_runs": len(runs),
        "completed_primary_protocol_runs": len(runs),
        "planned_tags": expected_tags,
        "completed_tags": completed_tags,
        "missing_tags": missing_tags,
        "all_planned_runs_complete": not missing_tags,
        "correlations": correlation_results, "detection_analysis": detection,
        "conventional_baseline_detection": conventional,
        "limitations": ["Condition-level correlations are descriptive and use related controlled variants.", "Three seeds quantify optimizer/training variability, not dataset-to-dataset generalization.", "A 10% trusted audit is a deterministic sample and is not equivalent to a full oracle."],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "statistical_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, rows in (("model_runs.csv", runs), ("condition_summary.csv", summary)):
        if rows:
            with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"completed": len(runs), "planned": len(expected_tags), "groups": len(summary), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
