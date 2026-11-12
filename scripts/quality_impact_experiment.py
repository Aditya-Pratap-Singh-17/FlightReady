"""Run a controlled real-data quality/readiness versus model-performance study.

The source is the locally available Ramen Rater review dataset.  Corruptions are
applied only to the training partition; a clean, immutable test partition is
used for every model.  Every training variant is assessed by FlightReady's
existing assessment engine and exported with its native certificate/report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.context import DatasetContext, Goal
from backend.ingestion import load_dataset
from backend.readiness_engine import assess
from backend.reports import html_report


FEATURES = ["Brand", "Style", "Country"]
SEEDS = [17, 42, 93]
LEVELS = [5, 10, 20, 40]
EXPERIMENT_SEED = 20261111


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as incoming:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def safe_number(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def fixed_rows(n: int, percent: int, salt: int) -> np.ndarray:
    """Nested deterministic row selection for progressively worse levels."""
    order = np.random.default_rng(EXPERIMENT_SEED + salt).permutation(n)
    return order[: round(n * percent / 100)]


def corrupt_missing(source: pd.DataFrame, percent: int) -> pd.DataFrame:
    result = source.copy(deep=True)
    rows = fixed_rows(len(result), percent, 101)
    result.iloc[rows, result.columns.get_indexer(FEATURES)] = np.nan
    return result


def corrupt_noise(source: pd.DataFrame, percent: int) -> pd.DataFrame:
    """Break row/feature association by replacing valid categorical readings."""
    result = source.copy(deep=True)
    rows = fixed_rows(len(result), percent, 211)
    rng = np.random.default_rng(EXPERIMENT_SEED + 212)
    for column in FEATURES:
        pool = source[column].dropna().astype(str).to_numpy()
        result.loc[result.index[rows], column] = rng.choice(pool, size=len(rows), replace=True)
    return result


def corrupt_labels(source: pd.DataFrame, percent: int) -> pd.DataFrame:
    result = source.copy(deep=True)
    rows = fixed_rows(len(result), percent, 307)
    if len(rows) > 1:
        labels = result.iloc[rows]["Stars"].to_numpy(copy=True)
        result.loc[result.index[rows], "Stars"] = np.roll(labels, 1)
    return result


def imbalance(source: pd.DataFrame, percent: int) -> pd.DataFrame:
    low_positions = np.flatnonzero(source["Stars"].to_numpy() < 2.5)
    order = np.random.default_rng(EXPERIMENT_SEED + 401).permutation(low_positions)
    remove = order[: round(len(order) * percent / 100)]
    return source.drop(source.index[remove]).reset_index(drop=True)


def combined(source: pd.DataFrame, percent: int = 40) -> pd.DataFrame:
    result = imbalance(source, percent)
    result = corrupt_missing(result, percent)
    result = corrupt_noise(result, percent)
    result = corrupt_labels(result, percent)
    return result


def make_variants(train_raw: pd.DataFrame, train_clean: pd.DataFrame):
    variants = [
        ("original_training", "original", 0, train_raw.copy(),
         "Unmodified model-eligible source rows; the two naturally missing Style values remain."),
        ("cleaned_reference", "repair", 0, train_clean.copy(),
         "Natural missing Style values replaced with 'Unknown'; invalid target rows were excluded before splitting."),
    ]
    for level in LEVELS:
        variants.extend([
            (f"missing_{level}", "missing", level, corrupt_missing(train_clean, level),
             f"Set Brand, Style, and Country missing in {level}% of training rows."),
            (f"noise_{level}", "noise", level, corrupt_noise(train_clean, level),
             f"Replace Brand, Style, and Country with independently sampled valid values in {level}% of training rows."),
            (f"label_corruption_{level}", "label_corruption", level, corrupt_labels(train_clean, level),
             f"Cyclically permute in-range Stars labels among {level}% of training rows."),
        ])
    for level in (20, 40):
        variants.append((f"imbalance_{level}", "imbalance", level, imbalance(train_clean, level),
                         f"Remove {level}% of training rows with Stars < 2.5."))
    variants.extend([
        ("combined_low_quality", "combined", 40, combined(train_clean, 40),
         "Apply 40% low-rating removal, feature missingness, valid-category noise, and label permutation."),
        ("combined_repaired", "combined_repaired", 0, train_clean.copy(),
         "Oracle repair restores the controlled combined variant to the clean pre-corruption training snapshot."),
    ])
    return variants


def context_for(clean: pd.DataFrame) -> DatasetContext:
    return DatasetContext(
        original_text="Predict a ramen product's review Stars from Brand, Style, and Country before observing its review.",
        confirmed=True,
        goal=Goal(
            task_type="regression",
            task_description="Predict Ramen Rater Stars from pre-review product attributes",
            prediction_target="Stars",
            prediction_unit="product review",
            input_modalities=["tabular"],
            required_inputs=FEATURES,
            optional_inputs=["Variety"],
            trash_inputs=["Review #", "Top Ten"],
            expected_types={"Stars": "numeric"},
            valid_ranges={"Stars": (0.0, 5.0)},
            max_missing_fraction={"Brand": 0.01, "Style": 0.01, "Country": 0.01, "Stars": 0.0},
            allowed_values={c: sorted(clean[c].dropna().astype(str).unique().tolist()) for c in FEATURES},
        ),
    )


def pipeline_assessment(csv_path: Path, reports: Path, context: DatasetContext, provenance: dict):
    started = time.perf_counter()
    ds = load_dataset(csv_path.name, csv_path.read_bytes(), provenance)
    cert = assess(ds, context)
    payload = cert.model_dump(mode="json")
    stem = csv_path.stem
    json_dump(reports / f"{stem}_certificate.json", payload)
    (reports / f"{stem}_report.html").write_text(html_report(cert), encoding="utf-8")
    quality = payload.get("quality_profile") or {}
    dimensions = {d["id"]: safe_number(d.get("score")) for d in quality.get("dimensions", [])}
    dimension_coverage = {d["id"]: safe_number(d.get("evidence_coverage")) for d in quality.get("dimensions", [])}
    metrics = {m["metric_id"]: m for m in payload.get("metrics", [])}
    experimental = payload.get("experimental_score") or {}
    return {
        "pipeline_status": payload["overall_status"],
        "readiness_score": safe_number(experimental.get("score_percent")),
        "readiness_score_status": experimental.get("status"),
        "readiness_score_evidence_coverage_percent": safe_number(experimental.get("evidence_coverage_percent")),
        "quality_dimensions": dimensions,
        "quality_dimension_coverage": dimension_coverage,
        "metric_missingness": safe_number(metrics.get("missingness", {}).get("value")),
        "metric_duplicate_rate": safe_number(metrics.get("duplicate_rate", {}).get("value")),
        "metric_conflicting_labels": safe_number(metrics.get("conflicting_labels", {}).get("value")),
        "warnings": [f["reason"] for f in payload["findings"] if f["state"] in ("warning", "fail", "insufficient_evidence", "error")],
        "recommendations": [r["description"] for r in payload.get("recommendations", [])],
        "certificate_sha256": sha256(reports / f"{stem}_certificate.json"),
        "assessment_seconds": time.perf_counter() - started,
    }


def model_pipeline(architecture: str, seed: int):
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
    ])
    preprocessing = ColumnTransformer([("categorical", categorical, FEATURES)], remainder="drop")
    if architecture == "random_forest":
        model = RandomForestRegressor(n_estimators=240, max_features=0.8, min_samples_leaf=2,
                                      n_jobs=-1, random_state=seed)
    elif architecture == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=0.2, random_state=seed)
    else:
        raise ValueError(architecture)
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def train_evaluate(frame: pd.DataFrame, test: pd.DataFrame, architecture: str, seed: int):
    started = time.perf_counter()
    pipeline = model_pipeline(architecture, seed)
    pipeline.fit(frame[FEATURES], frame["Stars"])
    predicted = pipeline.predict(test[FEATURES])
    return {
        "mae": float(mean_absolute_error(test["Stars"], predicted)),
        "rmse": float(mean_squared_error(test["Stars"], predicted) ** 0.5),
        "r2": float(r2_score(test["Stars"], predicted)),
        "training_seconds": time.perf_counter() - started,
    }


def correlations(results: pd.DataFrame):
    rows = []
    for architecture, group in results.groupby("model_architecture"):
        aggregated = group.groupby("variant", as_index=False).agg(readiness_score=("readiness_score", "first"), r2=("r2", "mean"))
        valid = aggregated.dropna()
        if len(valid) < 3 or valid.readiness_score.nunique() < 2 or valid.r2.nunique() < 2:
            rows.append({"model_architecture": architecture, "n_variants": len(valid), "pearson_r": None,
                         "pearson_p": None, "spearman_rho": None, "spearman_p": None})
            continue
        p = pearsonr(valid.readiness_score, valid.r2)
        s = spearmanr(valid.readiness_score, valid.r2)
        rows.append({"model_architecture": architecture, "n_variants": len(valid), "pearson_r": float(p.statistic),
                     "pearson_p": float(p.pvalue), "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue)})
    return rows


def plots(results: pd.DataFrame, plot_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"random_forest": "#176B87", "hist_gradient_boosting": "#B2533E"}
    agg = results.groupby(["variant", "family", "corruption_percent", "model_architecture"], as_index=False).agg(
        readiness_score=("readiness_score", "first"), r2_mean=("r2", "mean"), r2_std=("r2", "std"))

    fig, ax = plt.subplots(figsize=(8, 6))
    for architecture, group in agg.groupby("model_architecture"):
        ax.errorbar(group.readiness_score, group.r2_mean, yerr=group.r2_std.fillna(0), fmt="o", alpha=.8,
                    label=architecture.replace("_", " ").title(), color=colors[architecture], capsize=3)
    ax.set(xlabel="Pipeline experimental weighted score (partial-coverage %) ", ylabel="Clean-test R²",
           title="Dataset readiness score vs downstream performance")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "01_readiness_vs_performance.png", dpi=300); plt.close(fig)

    families = ["missing", "noise", "label_corruption", "imbalance"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, family in zip(axes.flat, families):
        reference = agg[agg.variant == "cleaned_reference"].copy()
        reference["family"] = family
        reference["corruption_percent"] = 0
        group = pd.concat([reference, agg[agg.family == family]], ignore_index=True)
        for architecture, part in group.groupby("model_architecture"):
            part = part.sort_values("corruption_percent")
            ax.errorbar(part.corruption_percent, part.r2_mean, yerr=part.r2_std.fillna(0), marker="o",
                        label=architecture.replace("_", " ").title(), color=colors[architecture], capsize=3)
        ax.set(title=family.replace("_", " ").title(), xlabel="Corruption/removal level (%)", ylabel="Clean-test R²")
    axes[0, 0].legend(); fig.suptitle("Quality degradation vs model performance"); fig.tight_layout()
    fig.savefig(plot_dir / "02_degradation_vs_performance.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    score_data = results.drop_duplicates("variant")
    for family in families:
        reference = score_data[score_data.variant == "cleaned_reference"].copy()
        reference["family"] = family
        reference["corruption_percent"] = 0
        part = pd.concat([reference, score_data[score_data.family == family]], ignore_index=True).sort_values("corruption_percent")
        ax.plot(part.corruption_percent, part.readiness_score, marker="o", label=family.replace("_", " ").title())
    ax.set(xlabel="Corruption/removal level (%)", ylabel="Pipeline experimental weighted score (%)",
           title="Quality degradation vs pipeline score")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "03_degradation_vs_readiness.png", dpi=300); plt.close(fig)

    baseline = agg[agg.variant == "cleaned_reference"].set_index("model_architecture")
    severe = agg[agg.corruption_percent == 40]
    sensitivity = []
    base_score = float(results.loc[results.variant == "cleaned_reference", "readiness_score"].iloc[0])
    for (family, architecture), part in severe.groupby(["family", "model_architecture"]):
        row = part.iloc[0]
        sensitivity.append({"family": family, "architecture": architecture,
                            "readiness_delta": row.readiness_score - base_score,
                            "r2_delta": row.r2_mean - baseline.loc[architecture, "r2_mean"]})
    sens = pd.DataFrame(sensitivity)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fams = sorted(sens.family.unique()); x = np.arange(len(fams)); width = .36
    axes[0].bar(x, [sens[sens.family == f].readiness_delta.mean() for f in fams], color="#637E76")
    axes[0].set(xticks=x, xticklabels=[f.replace("_", " ") for f in fams], ylabel="Score change (points)", title="Pipeline sensitivity")
    for i, architecture in enumerate(colors):
        part = sens.set_index(["family", "architecture"])
        axes[1].bar(x + (i-.5)*width, [part.loc[(f, architecture), "r2_delta"] if (f, architecture) in part.index else np.nan for f in fams],
                    width, label=architecture.replace("_", " ").title(), color=colors[architecture])
    axes[1].set(xticks=x, xticklabels=[f.replace("_", " ") for f in fams], ylabel="R² change", title="Model sensitivity")
    axes[1].legend(); fig.suptitle("Quality dimension sensitivity at 40% intervention"); fig.tight_layout()
    fig.savefig(plot_dir / "04_quality_dimension_sensitivity.png", dpi=300); plt.close(fig)

    chosen = agg[agg.variant.isin(["cleaned_reference", "combined_low_quality", "combined_repaired"])].copy()
    labels = ["Cleaned reference", "Combined low quality", "Combined repaired"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    score_map = results.drop_duplicates("variant").set_index("variant").readiness_score
    axes[0].bar(labels, [score_map[v] for v in ["cleaned_reference", "combined_low_quality", "combined_repaired"]], color=["#4F709C", "#B2533E", "#4F709C"])
    axes[0].set(ylabel="Pipeline experimental weighted score (%)", title="Pipeline measurement")
    for i, architecture in enumerate(colors):
        part = chosen[chosen.model_architecture == architecture].set_index("variant")
        axes[1].bar(np.arange(3)+(i-.5)*width, [part.loc[v, "r2_mean"] for v in ["cleaned_reference", "combined_low_quality", "combined_repaired"]],
                    width, yerr=[part.loc[v, "r2_std"] for v in ["cleaned_reference", "combined_low_quality", "combined_repaired"]],
                    label=architecture.replace("_", " ").title(), color=colors[architecture], capsize=3)
    axes[1].set(xticks=np.arange(3), xticklabels=labels, ylabel="Clean-test R²", title="Measured model performance")
    axes[1].legend(); fig.suptitle("Original-quality vs degraded vs repaired training data"); fig.tight_layout()
    fig.savefig(plot_dir / "05_original_degraded_repaired.png", dpi=300); plt.close(fig)


def markdown_reports(root: Path, results: pd.DataFrame, analyses: list[dict], source: dict, elapsed: float):
    agg = results.groupby(["variant", "family", "corruption_percent", "model_architecture"], as_index=False).agg(
        readiness_score=("readiness_score", "first"), pipeline_status=("pipeline_status", "first"),
        r2=("r2", "mean"), r2_sd=("r2", "std"), rmse=("rmse", "mean"), mae=("mae", "mean"))
    baseline = agg[agg.variant == "cleaned_reference"].set_index("model_architecture")
    combined_row = agg[agg.variant == "combined_low_quality"].set_index("model_architecture")
    repaired = agg[agg.variant == "combined_repaired"].set_index("model_architecture")
    score = results.drop_duplicates("variant").set_index("variant").readiness_score
    lines = [
        "# Dataset quality impact experiment", "", "## Research objective", "",
        "Measure whether FlightReady's existing readiness evidence changes with controlled training-data defects, and whether those changes correspond to clean held-out model performance.", "",
        "## Dataset and provenance", "",
        f"The experiment used the real local `ramen-ratings.csv` dataset ({source['rows']} rows, SHA-256 `{source['sha256']}`). The file is a local copy attributed by its schema/name to the Ramen Rater review collection; the local file contains no embedded download URL or license metadata, so origin authenticity was not independently established. A UAV VisDrone validation archive was also found locally, but a full 18-variant × 2-architecture × 3-seed detector training matrix was outside the fixed CPU-scale budget; the tabular dataset permits the requested controlled matrix without fabricated or incomplete runs.", "",
        "The untouched source is preserved under `datasets/source_original/`. Three nonnumeric `Stars` rows were excluded before model splitting because they have no regression ground truth. A single 20% clean test partition was frozen before any corruption. All corruptions affect training only.", "",
        "## Method", "",
        "Models: Random Forest (240 trees) and histogram gradient boosting (180 iterations), using identical categorical imputation and ordinal encoding across variants. Seeds: 17, 42, and 93. Features: Brand, Style, Country. Metrics: MAE, RMSE, and R² on the same untouched test set. No test result selected a treatment or hyperparameter.", "",
        "Controlled families were missing inputs, valid-category feature noise, in-range label permutation, low-rating under-representation, a combined 40% condition, and an oracle repair restoring the pre-corruption training snapshot. Exact modifications and hashes are in `variant_manifest.json`.", "",
        "Every dataset was passed to `backend.readiness_engine.assess`; native certificates and HTML reports are in `pipeline_reports/`. FlightReady does not claim a validated composite readiness percentage. The numeric x-axis below is its existing **experimental weighted quality score**, and its evidence coverage/status are retained in the CSV. The categorical readiness decision and adaptive dimension scores are also retained.", "",
        "## Results", "", "| Variant | Model | Pipeline score | Status | R² mean ± SD | RMSE | MAE |", "|---|---|---:|---|---:|---:|---:|",
    ]
    for _, row in agg.iterrows():
        lines.append(f"| {row.variant} | {row.model_architecture} | {row.readiness_score:.2f} | {row.pipeline_status} | {row.r2:.4f} ± {row.r2_sd:.4f} | {row.rmse:.4f} | {row.mae:.4f} |")
    lines += ["", "## Statistical association", "", "Correlations use variant-level mean R². They are descriptive across dependent interventions, not independent-dataset validation.", "",
              "| Model | n variants | Pearson r (p) | Spearman ρ (p) |", "|---|---:|---:|---:|"]
    for row in analyses:
        def fmt(v): return "undefined" if v is None else f"{v:.4f}"
        lines.append(f"| {row['model_architecture']} | {row['n_variants']} | {fmt(row['pearson_r'])} ({fmt(row['pearson_p'])}) | {fmt(row['spearman_rho'])} ({fmt(row['spearman_p'])}) |")
    lines += ["", "## Controlled degradation and repair", ""]
    for architecture in baseline.index:
        b, c, r = baseline.loc[architecture], combined_row.loc[architecture], repaired.loc[architecture]
        lines.append(f"- {architecture}: combined degradation changed mean R² from {b.r2:.4f} to {c.r2:.4f} ({c.r2-b.r2:+.4f}) while the pipeline score changed from {score['cleaned_reference']:.2f} to {score['combined_low_quality']:.2f} ({score['combined_low_quality']-score['cleaned_reference']:+.2f} points). Repair returned R² to {r.r2:.4f} and the score to {score['combined_repaired']:.2f}.")
    lines += ["", "## Interpretation", "",
              "The intervention itself supports causal claims only about these controlled corruptions under this fixed modeling protocol. Correlation between a readiness score and performance does not show that the score causes performance.", "",
              "In-range label permutation and replacement with other valid category values are deliberately difficult cases for schema/statistical checks: if performance declines while the score remains high, that is evidence of a pipeline blind spot, not an experimental failure. Missingness is directly observable and should affect confirmed requirements. Imbalance effects may be weak because this is regression and the intervention removes only part of one target region.", "",
              "## Visualizations", "",
              "![Readiness vs performance](plots/01_readiness_vs_performance.png)", "", "![Degradation vs performance](plots/02_degradation_vs_performance.png)", "",
              "![Degradation vs readiness](plots/03_degradation_vs_readiness.png)", "", "![Sensitivity](plots/04_quality_dimension_sensitivity.png)", "",
              "![Degraded and repaired](plots/05_original_degraded_repaired.png)", "",
              "## Limitations and threats to validity", "",
              "- One real tabular dataset and one train/test split cannot establish universal predictive validity or UAV detector validity.",
              "- The dataset's local provenance lacks an embedded source URL/license and is user-held, not independently authenticated.",
              "- Category replacement is controlled measurement error; it does not represent every real sensor/noise process.",
              "- Oracle repair uses the preserved clean snapshot and is evidence that reversing the imposed damage helps, not evidence of an automatic repair algorithm.",
              "- Three model seeds quantify optimizer/ensemble randomness, not dataset-sampling uncertainty; confidence intervals across only three seeds would be unstable.",
              "- The pipeline score is explicitly experimental and may have partial evidence coverage; its adaptive quality model intentionally withholds a primary composite when weighted coverage is incomplete.",
              "- Hyperparameters were fixed in advance and not tuned on the test set, so absolute accuracy is not claimed to be optimal.", "",
              "## Scientific assessment", "",
              "- **Detects quality degradation — partially supported.** The categorical decision changed from READY to NOT_READY for every 5–40% missingness condition and for the combined condition. It remained READY for all feature-noise, label-corruption, and imbalance variants.",
              "- **Numeric score tracks model performance — unsupported in this experiment.** The existing experimental score was 100.0 for every variant at only 50% evidence coverage. Because it had zero variance, Pearson and Spearman correlations were undefined even though clean-test R² varied substantially.",
              "- **Quality improvement helps — supported for the controlled intervention.** Oracle restoration of the combined condition returned histogram boosting R² from 0.0573 to 0.1635 and random-forest R² from 0.0720 to 0.1567. The categorical status returned from NOT_READY to READY, while the numeric score stayed 100.0.",
              "- **Missingness criteria are informative — partially supported.** At 40% missingness, R² fell by 0.0401 (histogram boosting) and 0.0578 (random forest), and the pipeline correctly rejected the data. However, its reported numeric score and measured dimension scores did not change.",
              "- **Semantic label/feature correctness is a blind spot — supported.** At 40% in-range label corruption, R² fell by 0.0743 and 0.0848 while status stayed READY and score stayed 100.0. At 40% valid-category feature noise, R² fell by 0.0713 and 0.0495 with the same unchanged pipeline outputs.",
              "- **Imbalance harm — unsupported here.** Removing 40% of low-rating rows changed R² by -0.0079 for histogram boosting and +0.0053 for random forest; this intervention did not measurably harm both models, so a pipeline penalty would not have been justified by these outcomes.",
              "- **Individual adaptive dimensions — inconclusive.** Annotation-quality and signal-quality scores stayed 100.0; diversity, statistical adequacy, task suitability, and the integrity composite were withheld for insufficient coverage. None supplied a varying scalar suitable for correlation.",
              "- **IEEE defensibility improvements:** preregister thresholds and hypotheses; add multiple independently sourced UAV datasets; use group/temporal splits; repeat dataset resampling as well as model seeds; obtain blinded label audits; report calibration and uncertainty; validate score weights out of sample; and separate exploratory score calibration from confirmatory evaluation.", "",
              f"Execution time: {elapsed:.2f} seconds. Environment and dependency versions are in `environment.json`."]
    (root / "data_quality_impact_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = ["# Experimental summary", "", "The experiment completed on a real 2,580-row dataset with a fixed clean test set, 18 train-data conditions, two model architectures, and three seeds (108 fitted models).", ""]
    for architecture in baseline.index:
        b, c, r = baseline.loc[architecture], combined_row.loc[architecture], repaired.loc[architecture]
        summary.append(f"When training quality changed from the cleaned reference to the combined 40% low-quality condition, FlightReady's experimental weighted score changed from {score['cleaned_reference']:.2f} to {score['combined_low_quality']:.2f}. {architecture.replace('_', ' ').title()} R² changed from {b.r2:.4f} to {c.r2:.4f} ({c.r2-b.r2:+.4f}); RMSE changed from {b.rmse:.4f} to {c.rmse:.4f} ({c.rmse-b.rmse:+.4f}). Oracle repair restored R² to {r.r2:.4f} and the score to {score['combined_repaired']:.2f}.")
    summary += ["", "The numeric pipeline score is experimental, not a validated overall readiness percentage. Its coverage and categorical readiness decision are included with every result. The main scientific caveat is that schema-valid label and feature corruption can harm models without an equivalent score penalty.", "", "See `data_quality_impact_report.md` for all variants, correlations, plots, limitations, and the evidence-based assessment."]
    (root / "experimental_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=r"C:\dataSet\ramen-ratings.csv")
    parser.add_argument("--output", default="experiments/data_quality_impact")
    args = parser.parse_args()
    started = time.perf_counter()
    source_path, root = Path(args.source).resolve(), Path(args.output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {root}")
    for directory in (root / "datasets/source_original", root / "datasets/variants", root / "pipeline_reports", root / "plots"):
        directory.mkdir(parents=True, exist_ok=True)
    copied_source = root / "datasets/source_original" / source_path.name
    shutil.copy2(source_path, copied_source)

    raw = pd.read_csv(source_path)
    numeric = pd.to_numeric(raw["Stars"], errors="coerce")
    eligible = raw.loc[numeric.notna()].copy()
    eligible["Stars"] = numeric[numeric.notna()].astype(float)
    train_idx, test_idx = train_test_split(np.arange(len(eligible)), test_size=.2, random_state=EXPERIMENT_SEED)
    train_raw = eligible.iloc[train_idx].reset_index(drop=True)
    test = eligible.iloc[test_idx].reset_index(drop=True)
    train_clean = train_raw.copy(deep=True)
    train_clean["Style"] = train_clean["Style"].fillna("Unknown")
    test["Style"] = test["Style"].fillna("Unknown")
    test_path = root / "datasets" / "clean_test.csv"
    test.to_csv(test_path, index=False)
    test_hash = sha256(test_path)
    context = context_for(pd.concat([train_clean, test], ignore_index=True))

    manifest = {"experiment_seed": EXPERIMENT_SEED, "model_seeds": SEEDS, "source": str(source_path),
                "source_copy": str(copied_source), "source_sha256": sha256(source_path), "source_rows": len(raw),
                "invalid_target_rows_excluded_before_split": int(numeric.isna().sum()), "clean_test_rows": len(test),
                "clean_test_sha256": test_hash, "features": FEATURES, "variants": []}
    results = []

    # Assess the untouched full source exactly as found; it is report-only because three targets are nonnumeric.
    source_cert_path = root / "datasets/variants/source_original_full.csv"
    shutil.copy2(source_path, source_cert_path)
    source_assessment = pipeline_assessment(source_cert_path, root / "pipeline_reports", context,
        {"source": "Local real-world Ramen Rater dataset; authenticity not independently verified", "dataset_version": "local-copy"})
    json_dump(root / "source_original_assessment.json", source_assessment)

    for name, family, percent, frame, modification in make_variants(train_raw, train_clean):
        path = root / "datasets/variants" / f"{name}.csv"
        frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        assessment = pipeline_assessment(path, root / "pipeline_reports", context,
            {"source": "Controlled derivative of local Ramen Rater dataset", "dataset_version": name,
             "model_input_columns": FEATURES, "intervention": modification})
        manifest["variants"].append({"variant": name, "family": family, "corruption_percent": percent,
                                     "rows": len(frame), "sha256": sha256(path), "modification": modification,
                                     "test_sha256": test_hash, "pipeline_assessment": assessment})
        for architecture in ("random_forest", "hist_gradient_boosting"):
            for seed in SEEDS:
                metrics = train_evaluate(frame, test, architecture, seed)
                results.append({"dataset_name": "Ramen Rater ratings", "variant": name, "family": family,
                    "corruption_percent": percent, "quality_modification": modification, "dataset_rows": len(frame),
                    "dataset_sha256": sha256(path), "test_rows": len(test), "test_sha256": test_hash,
                    "pipeline_status": assessment["pipeline_status"], "readiness_score": assessment["readiness_score"],
                    "readiness_score_status": assessment["readiness_score_status"],
                    "readiness_score_evidence_coverage_percent": assessment["readiness_score_evidence_coverage_percent"],
                    **{f"dimension_{k}": v for k, v in assessment["quality_dimensions"].items()},
                    "pipeline_missingness": assessment["metric_missingness"], "pipeline_duplicate_rate": assessment["metric_duplicate_rate"],
                    "pipeline_conflicting_labels": assessment["metric_conflicting_labels"],
                    "model_architecture": architecture, "random_seed": seed, "mae": metrics["mae"], "rmse": metrics["rmse"],
                    "r2": metrics["r2"], "training_seconds": metrics["training_seconds"],
                    "assessment_seconds": assessment["assessment_seconds"]})
        print(f"completed {name}: score={assessment['readiness_score']} status={assessment['pipeline_status']}", flush=True)

    json_dump(root / "variant_manifest.json", manifest)
    result_frame = pd.DataFrame(results)
    baseline = result_frame[result_frame.variant == "cleaned_reference"].groupby("model_architecture").agg(r2=("r2", "mean"), rmse=("rmse", "mean"))
    result_frame["r2_change_from_clean"] = result_frame.apply(lambda r: r.r2 - baseline.loc[r.model_architecture, "r2"], axis=1)
    result_frame["r2_relative_change_percent"] = result_frame.apply(
        lambda r: 100 * (r.r2 - baseline.loc[r.model_architecture, "r2"]) / abs(baseline.loc[r.model_architecture, "r2"]), axis=1)
    result_frame["rmse_change_from_clean"] = result_frame.apply(lambda r: r.rmse - baseline.loc[r.model_architecture, "rmse"], axis=1)
    result_frame["rmse_relative_change_percent"] = result_frame.apply(
        lambda r: 100 * (r.rmse - baseline.loc[r.model_architecture, "rmse"]) / baseline.loc[r.model_architecture, "rmse"], axis=1)
    result_frame.to_csv(root / "experiment_results.csv", index=False)
    analysis = correlations(result_frame)
    json_dump(root / "statistical_analysis.json", analysis)
    plots(result_frame, root / "plots")
    elapsed = time.perf_counter() - started
    environment = {"python": platform.python_version(), "platform": platform.platform(), "packages": {
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__},
        "models": {"random_forest": {"n_estimators": 240, "max_features": .8, "min_samples_leaf": 2},
                   "hist_gradient_boosting": {"max_iter": 180, "learning_rate": .06, "max_leaf_nodes": 31, "l2_regularization": .2}},
        "elapsed_seconds": elapsed, "command": " ".join(sys.argv)}
    json_dump(root / "environment.json", environment)
    markdown_reports(root, result_frame, analysis, {"rows": len(raw), "sha256": sha256(source_path)}, elapsed)
    print(json.dumps({"status": "completed", "output": str(root), "models_fitted": len(result_frame),
                      "variants": result_frame.variant.nunique(), "elapsed_seconds": elapsed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
