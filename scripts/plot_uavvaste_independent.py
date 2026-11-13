"""Generate the eight required publication figures from completed artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
QUALITY = json.loads((EXP / "quality_results" / "flightready_assessments.json").read_text(encoding="utf-8"))["variants"]
ANALYSIS = EXP / "analysis"
FIG = EXP / "figures"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 150, "savefig.dpi": 300})


def read_csv(name):
    path = ANALYSIS / name
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summaries = read_csv("condition_summary.csv")
    runs = read_csv("model_runs.csv")
    primary = [r for r in summaries if r["model"] == "yolo11n" and r["schedule"] == "fixed_epochs"]

    # 1. Readiness versus performance.
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for row in primary:
        c = row["condition"]
        score = QUALITY[c]["scorecards"]["full_oracle_assisted"]["primary_quality_score"]
        y = float(row["map50_95_mean"])
        lo = float(row["map50_95_ci95_low"]) if row["map50_95_ci95_low"] else y
        hi = float(row["map50_95_ci95_high"]) if row["map50_95_ci95_high"] else y
        ax.errorbar(score, y, yerr=[[y-lo], [hi-y]], fmt="o", color="#145DA0", capsize=3)
        ax.annotate(c.replace("_", " "), (score, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="FlightReady full-oracle quality score", ylabel="Test mAP@0.5:0.95", title="Readiness assessment and detector performance")
    ax.grid(alpha=.25)
    save(fig, "01_readiness_vs_performance.png")

    # 2. Severity curves.
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for family, prefix, color in (("Gaussian noise", "gaussian", "#D1495B"), ("Missing annotations", "missing_annotations", "#00798C")):
        points = []
        clean = next((r for r in primary if r["condition"] == "clean"), None)
        if clean: points.append((0, float(clean["map50_95_mean"])))
        for r in primary:
            if r["condition"].startswith(prefix + "_"):
                points.append((int(r["condition"].split("_")[-1]), float(r["map50_95_mean"])))
        for r in summaries:
            if r["model"] == "yolo11n" and r["condition"].startswith(prefix + "_") and (int(r["condition"].split("_")[-1]), float(r["map50_95_mean"])) not in points:
                points.append((int(r["condition"].split("_")[-1]), float(r["map50_95_mean"])))
        points.sort()
        if points: ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=family, color=color)
    ax.set(xlabel="Affected training images (%)", ylabel="Test mAP@0.5:0.95", title="Detector performance by degradation prevalence")
    ax.legend(); ax.grid(alpha=.25)
    save(fig, "02_performance_vs_severity.png")

    # 3. Metric sensitivity heatmap.
    variants = [v for v in QUALITY if v not in {"clean", "repaired"}]
    clean_metrics = {m["metric_id"]: m for m in QUALITY["clean"]["scorecards"]["full_oracle_assisted"]["metrics"]}
    metric_ids = list(clean_metrics)
    matrix = []
    for v in variants:
        current = {m["metric_id"]: m for m in QUALITY[v]["scorecards"]["full_oracle_assisted"]["metrics"]}
        matrix.append([(clean_metrics[mid]["score"] or 0) - (current[mid]["score"] or 0) for mid in metric_ids])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(metric_ids)), [m.replace("reference_", "ref ").replace("_", " ") for m in metric_ids], rotation=45, ha="right")
    ax.set_yticks(range(len(variants)), [v.replace("_", " ") for v in variants])
    ax.set_title("FlightReady metric sensitivity (score-point loss from clean)")
    fig.colorbar(image, ax=ax, label="Score-point loss")
    save(fig, "03_metric_sensitivity.png")

    # 4. Reference-free versus assisted detection.
    stat = json.loads((ANALYSIS / "statistical_analysis.json").read_text(encoding="utf-8"))
    modes = ["reference_free", "deterministic_10_percent_annotation_audit", "full_oracle_assisted"]
    values = [stat["detection_analysis"][m]["sensitivity"] for m in modes]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(["Reference-free", "10% trusted audit", "Full oracle"], values, color=["#9AA0A6", "#F4B942", "#2E8B57"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Defect-condition sensitivity"); ax.set_title("Trusted evidence improves controlled-defect detection")
    for i, value in enumerate(values): ax.text(i, value + .025, f"{value:.0%}", ha="center")
    save(fig, "04_reference_evidence_detection.png")

    # 5. Evidence coverage.
    variants_all = list(QUALITY)
    free = [QUALITY[v]["scorecards"]["reference_free"]["evidence_coverage"] for v in variants_all]
    assisted = [QUALITY[v]["scorecards"]["full_oracle_assisted"]["evidence_coverage"] for v in variants_all]
    x = np.arange(len(variants_all)); width = .38
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(x-width/2, free, width, label="Reference-free", color="#9AA0A6")
    ax.bar(x+width/2, assisted, width, label="Full oracle", color="#2E8B57")
    ax.axhline(.8, color="#D1495B", ls="--", lw=1, label="Decision coverage threshold")
    ax.set_xticks(x, [v.replace("_", " ") for v in variants_all], rotation=45, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Weighted evidence coverage"); ax.set_title("Evidence coverage across conditions"); ax.legend(ncol=3, fontsize=8)
    save(fig, "05_evidence_coverage.png")

    # 6. Clean, combined, repaired.
    selected = [r for r in primary if r["condition"] in {"clean", "combined_40", "repaired"}]
    selected.sort(key=lambda r: ["clean", "combined_40", "repaired"].index(r["condition"]))
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    vals = [float(r["map50_95_mean"]) for r in selected]
    errors = [max(0, float(r["map50_95_ci95_high"]) - float(r["map50_95_mean"])) if r["map50_95_ci95_high"] else 0 for r in selected]
    ax.bar([r["condition"].replace("_", " ") for r in selected], vals, yerr=errors, capsize=4, color=["#145DA0", "#D1495B", "#2E8B57"][:len(vals)])
    ax.set(ylabel="Test mAP@0.5:0.95", title="Original, combined degradation, and repair")
    save(fig, "06_degraded_repaired_performance.png")

    # 7. FlightReady and conventional baseline detection.
    conventional = {r["variant"]: r["detected"] for r in stat["conventional_baseline_detection"]}
    defect = list(conventional)
    full_detected = set(stat["detection_analysis"]["full_oracle_assisted"]["detected"])
    free_detected = set(stat["detection_analysis"]["reference_free"]["detected"])
    mat = np.array([[v in free_detected, v in full_detected, conventional[v]] for v in defect], dtype=int)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["FlightReady\nreference-free", "FlightReady\nfull oracle", "Conventional\nchecks"])
    ax.set_yticks(range(len(defect)), [v.replace("_", " ") for v in defect])
    for i in range(len(defect)):
        for j in range(3): ax.text(j, i, "Detected" if mat[i,j] else "Missed", ha="center", va="center", fontsize=7, color="white" if mat[i,j] else "black")
    ax.set_title("Shared-capability defect detection comparison")
    save(fig, "07_method_detection_comparison.png")

    # 8. Seed variability.
    condition_order = [v for v in QUALITY if any(r["condition"] == v and r["model"] == "yolo11n" and r["schedule"] == "fixed_epochs" for r in runs)]
    data = [[float(r["map50_95"]) for r in runs if r["condition"] == c and r["model"] == "yolo11n" and r["schedule"] == "fixed_epochs"] for c in condition_order]
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    if data: ax.boxplot(data, tick_labels=[c.replace("_", " ") for c in condition_order], showmeans=True)
    ax.tick_params(axis="x", rotation=45); ax.set(ylabel="Test mAP@0.5:0.95", title="Performance variability across model seeds")
    save(fig, "08_seed_variability.png")
    print(json.dumps({"figures": 8, "directory": str(FIG)}))


if __name__ == "__main__":
    main()
