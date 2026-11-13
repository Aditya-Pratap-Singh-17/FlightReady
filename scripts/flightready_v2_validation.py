"""Reproducible FlightReady v1/v2 reassessment using preserved artifacts.

No model is trained by this script. Historical downstream measurements are
read from immutable CSV/JSON artifacts. Reference-assisted checks are labelled
as such and are also contrasted with a reference-free scorecard.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metrics.readiness_v2 import V2Metric, build_scorecard, fraction_metric

TAB = ROOT / "experiments" / "data_quality_impact"
VIS = ROOT / "experiments" / "visdrone_model_validation"
OUT = ROOT / "experiments" / "flightready_v2_validation"
PLOTS = OUT / "plots"
DOCS = ROOT / "docs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def js_divergence(a: pd.Series, b: pd.Series) -> float:
    ac, bc = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
    keys = sorted(set(ac.index) | set(bc.index))
    p = np.array([ac.get(k, 0) for k in keys], dtype=float)
    q = np.array([bc.get(k, 0) for k in keys], dtype=float)
    m = (p + q) / 2
    def kl(x):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / m[mask])))
    return (kl(p) + kl(q)) / 2


def score_metric(metric_id, dimension, label, score, raw, *, reference=False, limitation, gate=None, passed=None):
    return V2Metric(metric_id=metric_id, dimension=dimension, label=label, weight=1.0,
                    status="measured", score=float(max(0, min(100, score))), raw_value=raw,
                    normalization="documented direct criterion", evidence_type="trusted_experimental_reference" if reference else "observed",
                    reference_id="controlled-clean-oracle" if reference else None, limitation=limitation,
                    gate=gate, gate_passed=passed)


def unavailable(metric_id, dimension, label, limitation, *, gate=None):
    return V2Metric(metric_id=metric_id, dimension=dimension, label=label, weight=1.0,
                    status="unavailable", normalization="not computed without required evidence",
                    evidence_type="missing_reference", limitation=limitation, gate=gate)


def tabular_validation():
    manifest = json.loads((TAB / "variant_manifest.json").read_text(encoding="utf-8"))
    reference_path = TAB / "datasets" / "variants" / "cleaned_reference.csv"
    ref = pd.read_csv(reference_path).set_index("Review #", drop=False)
    historical = pd.read_csv(TAB / "experiment_results.csv")
    performance = historical.groupby(["variant", "model_architecture"], as_index=False)["r2"].mean()
    performance_all = historical.groupby("variant")["r2"].mean().to_dict()
    rows, details, integrity = [], {}, []
    for entry in manifest["variants"]:
        name = entry["variant"]
        path = TAB / "datasets" / "variants" / f"{name}.csv"
        observed_hash = sha256(path)
        integrity.append({"variant": name, "expected_sha256": entry["sha256"], "observed_sha256": observed_hash,
                          "valid": observed_hash == entry["sha256"]})
        frame = pd.read_csv(path).set_index("Review #", drop=False)
        common = frame.index.intersection(ref.index)
        features = ["Brand", "Style", "Country"]
        missing = float(frame[features].isna().sum().sum() / (len(frame) * len(features)))
        feature_disagreement = float(sum((frame.loc[common, c].fillna("<NA>") != ref.loc[common, c].fillna("<NA>")).sum() for c in features) / (len(common) * len(features)))
        label_disagreement = float((frame.loc[common, "Stars"] != ref.loc[common, "Stars"]).mean())
        coverage_loss = 1 - len(frame) / len(ref)
        duplicates = float(frame.duplicated().mean())
        target_js = js_divergence(frame["Stars"], ref["Stars"])
        assisted = [
            fraction_metric("cell_missingness", "integrity", "Required feature missingness", missing, limitation="Measures observed required feature cells only.", gate="block", maximum=.01),
            fraction_metric("duplicate_records", "integrity", "Exact duplicate records", duplicates, limitation="Exact repeats can be legitimate; record identity needs domain review."),
            fraction_metric("feature_reference_disagreement", "feature_quality", "Feature disagreement with preserved clean rows", feature_disagreement, evidence_type="trusted_experimental_reference", reference_id="cleaned_reference", limitation="Valid only because controlled variants preserve row identifiers; not available for an arbitrary dataset.", gate="warning", maximum=0),
            fraction_metric("label_reference_disagreement", "annotation_quality", "Target disagreement with preserved clean labels", label_disagreement, evidence_type="trusted_experimental_reference", reference_id="cleaned_reference", limitation="This is oracle-assisted experimental evidence, not semantic-error detection from dataset statistics.", gate="warning", maximum=0),
            fraction_metric("expected_record_loss", "coverage", "Missing expected records", coverage_loss, evidence_type="trusted_experimental_manifest", reference_id="variant_manifest", limitation="Requires an authoritative expected acquisition manifest.", gate="block", maximum=0),
            score_metric("target_distribution_similarity", "statistical_adequacy", "Target distribution similarity", 100*(1-target_js), target_js, reference=True, limitation="Jensen-Shannon similarity is descriptive and does not prove representativeness."),
        ]
        free = [assisted[0], assisted[1], unavailable("feature_reference_disagreement", "feature_quality", "Feature semantic correctness", "No trusted row-level reference.", gate="warning"),
                unavailable("label_reference_disagreement", "annotation_quality", "Target semantic correctness", "No trusted labels; distribution shape cannot prove correctness.", gate="warning"),
                unavailable("expected_record_loss", "coverage", "Acquisition completeness", "No authoritative expected-record manifest.", gate="block"),
                unavailable("target_distribution_similarity", "statistical_adequacy", "Deployment distribution similarity", "No relevant target-population reference.")]
        card, free_card = build_scorecard(assisted), build_scorecard(free)
        original = entry["pipeline_assessment"]
        row = {"variant": name, "family": entry["family"], "corruption_percent": entry["corruption_percent"],
               "original_score": original["readiness_score"], "improved_score": card["measured_quality_score"],
               "original_status": original["pipeline_status"], "improved_status": card["readiness_decision"],
               "original_coverage_percent": original["readiness_score_evidence_coverage_percent"],
               "improved_coverage_percent": 100*card["evidence_coverage"],
               "reference_free_status": free_card["readiness_decision"], "reference_free_coverage_percent": 100*free_card["evidence_coverage"],
               "mean_r2": performance_all.get(name), "missing_fraction": missing, "feature_disagreement": feature_disagreement,
               "label_disagreement": label_disagreement, "record_loss": coverage_loss, "target_js_divergence": target_js}
        rows.append(row); details[name] = {"reference_assisted": card, "reference_free": free_card}
    return pd.DataFrame(rows), performance, details, integrity


def label_records(root: Path):
    labels = root / "labels" / "train"
    records, invalid = {}, 0
    for path in labels.glob("*.txt"):
        parsed = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            bits = line.split()
            try:
                values = [float(x) for x in bits]
                if len(values) != 5 or int(values[0]) != values[0] or not all(math.isfinite(x) for x in values) or min(values[3:]) <= 0 or min(values[1:3]) < 0 or max(values[1:3]) > 1:
                    invalid += 1; continue
                parsed.append((int(values[0]), *values[1:]))
            except ValueError:
                invalid += 1
        records[path.name] = parsed
    return records, invalid


def annotation_comparison(records, reference):
    omitted = wrong = geometry = compared = 0
    classes = Counter()
    for name, ref_rows in reference.items():
        rows = records.get(name, [])
        omitted += max(0, len(ref_rows)-len(rows))
        for left, right in zip(rows, ref_rows):
            compared += 1; classes[left[0]] += 1
            wrong += left[0] != right[0]
            geometry += any(abs(a-b) > 1e-6 for a, b in zip(left[1:], right[1:]))
    total_ref = sum(map(len, reference.values()))
    return {"box_omission_fraction": omitted/total_ref if total_ref else 0,
            "class_disagreement_fraction": wrong/compared if compared else 0,
            "geometry_disagreement_fraction": geometry/compared if compared else 0,
            "compared_boxes": compared, "reference_boxes": total_ref, "class_counts": dict(classes)}


def image_features(path: Path):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    lap = -4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
    smooth = np.asarray(Image.fromarray(np.uint8(gray*255)).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)/255
    residual = float(np.median(np.abs(gray-smooth)))
    boundary = np.abs(np.diff(gray, axis=1))[:,7::8]
    return {"contrast": float(gray.std()), "laplacian_variance": float(lap.var()),
            "high_frequency_mad": residual, "blockiness": float(boundary.mean()) if boundary.size else 0.0}


def sampled_image_comparison(root: Path, clean: Path, seed=20261111, n=48):
    candidates = sorted((root / "images" / "train").glob("*.jpg"))
    ranked = sorted(candidates, key=lambda p: hashlib.sha256(f"{seed}:{p.name}".encode()).digest())[:n]
    values, errors = [], 0
    for path in ranked:
        clean_path = clean / "images" / "train" / path.name
        if not clean_path.exists(): continue
        try:
            a, b = image_features(path), image_features(clean_path)
            with Image.open(path) as ia, Image.open(clean_path) as ib:
                aa = np.asarray(ia.convert("RGB").resize((128,72)), dtype=np.float32)
                bb = np.asarray(ib.convert("RGB").resize((128,72)), dtype=np.float32)
            values.append({**{f"{k}_ratio": a[k]/b[k] if b[k] else 1 for k in a}, "pixel_mae": float(np.abs(aa-bb).mean()/255)})
        except Exception:
            errors += 1
    means = {k: float(np.mean([v[k] for v in values])) for k in values[0]} if values else {}
    return {"sample_size": len(values), "decode_errors": errors, **means}


def controlled_duplicate_fraction(root: Path):
    """Verify generated duplicate files byte-for-byte against their named source."""
    images = root / "images" / "train"
    total = sum(1 for _ in images.glob("*.jpg"))
    verified = 0
    candidates = 0
    for path in images.glob("duplicate_*.jpg"):
        candidates += 1
        parts = path.name.split("_", 2)
        source = images / parts[2] if len(parts) == 3 else None
        if source and source.exists() and sha256(path) == sha256(source):
            verified += 1
    return verified / total if total else 0.0, {"filename_candidates": candidates, "byte_verified_duplicates": verified, "image_population": total}


def visdrone_validation():
    manifest = json.loads((VIS / "variant_manifest.json").read_text(encoding="utf-8"))
    clean_root = Path(manifest["variants"][0]["root"])
    clean_labels, clean_invalid = label_records(clean_root)
    historical = pd.read_csv(VIS / "results" / "performance_comparison.csv")
    hist = historical.set_index("variant").to_dict("index")
    rows, details, integrity = [], {}, []
    clean_class = Counter(r[0] for values in clean_labels.values() for r in values)
    for entry in manifest["variants"]:
        root = Path(entry["root"])
        records, invalid = label_records(root)
        ann = annotation_comparison(records, clean_labels)
        image_count = sum(1 for _ in (root / "images" / "train").glob("*.jpg"))
        expected_loss = max(0.0, 1-image_count/manifest["original_images"])
        image_sample = sampled_image_comparison(root, clean_root)
        duplicate_fraction, duplicate_evidence = controlled_duplicate_fraction(root)
        class_series = pd.Series({str(k):v for k,v in ann["class_counts"].items()})
        clean_series = pd.Series({str(k):v for k,v in clean_class.items()})
        # Expand counts so the shared categorical helper remains transparent.
        class_js = js_divergence(pd.Series(np.repeat(class_series.index, class_series.values.astype(int))), pd.Series(np.repeat(clean_series.index, clean_series.values.astype(int)))) if len(class_series) else 1.0
        pixel_change = image_sample.get("pixel_mae", 0.0)
        # Paired clean-reference fidelity detects controlled image changes; the
        # individual ratios remain visible and are never interpreted alone.
        metrics = [
            fraction_metric("invalid_annotations", "annotation_quality", "Invalid annotation syntax", invalid/max(1, sum(map(len, records.values()))+invalid), limitation="Syntax only; valid syntax is not semantic correctness.", gate="block", maximum=0),
            fraction_metric("box_omission", "annotation_quality", "Missing boxes versus trusted clean reference", ann["box_omission_fraction"], evidence_type="trusted_experimental_reference", reference_id="clean", limitation="Reference-assisted experimental check; cannot infer visible-object completeness without trusted annotations.", gate="warning", maximum=0),
            fraction_metric("class_disagreement", "annotation_quality", "Wrong classes versus trusted clean reference", ann["class_disagreement_fraction"], evidence_type="trusted_experimental_reference", reference_id="clean", limitation="Reference-assisted; model disagreement alone would not prove an error.", gate="warning", maximum=0),
            fraction_metric("geometry_disagreement", "annotation_quality", "Box geometry disagreement", ann["geometry_disagreement_fraction"], evidence_type="trusted_experimental_reference", reference_id="clean", limitation="Any change is reported; task-appropriate localization tolerance remains context dependent.", gate="warning", maximum=0),
            fraction_metric("expected_image_loss", "coverage", "Missing expected training images", expected_loss, evidence_type="trusted_experimental_manifest", reference_id="variant_manifest", limitation="Requires authoritative expected acquisition count.", gate="block", maximum=0),
            score_metric("class_distribution_similarity", "coverage", "Class distribution similarity", 100*(1-class_js), class_js, reference=True, limitation="Distribution similarity cannot prove per-object label correctness."),
            score_metric("paired_pixel_fidelity", "signal_quality", "Paired image fidelity", 100*(1-min(1,pixel_change)), pixel_change, reference=True, limitation="Pixel differences can be legitimate; this is a controlled-reference change detector, not perceptual quality."),
            fraction_metric("exact_duplicate_images", "integrity", "Byte-verified duplicate images", duplicate_fraction, limitation="Controlled duplicate filenames are verified against source bytes; legitimate repeats still require domain review.", gate="warning", maximum=0),
        ]
        metrics[-2] = metrics[-2].model_copy(update={"gate":"warning", "gate_passed":pixel_change == 0})
        card = build_scorecard(metrics)
        free = build_scorecard([metrics[0], unavailable("box_omission", "annotation_quality", "Visible-object completeness", "Trusted annotations absent.", gate="warning"),
                                unavailable("class_disagreement", "annotation_quality", "Semantic class correctness", "Trusted annotations absent.", gate="warning"),
                                unavailable("geometry_disagreement", "annotation_quality", "Semantic localization correctness", "Trusted annotations absent.", gate="warning"),
                                unavailable("expected_image_loss", "coverage", "Acquisition completeness", "Expected count absent.", gate="block"),
                                unavailable("class_distribution_similarity", "coverage", "Required class distribution", "Deployment reference absent."),
                                score_metric("image_quality_proxies", "signal_quality", "Observed image quality proxies", 100*math.exp(-abs(math.log(max(image_sample.get("contrast_ratio",1),1e-6)))-abs(math.log(max(image_sample.get("laplacian_variance_ratio",1),1e-6)))), image_sample, limitation="Reference ratios are diagnostic proxies; texture, scale and noise confound them."),
                                metrics[-1]])
        old = hist.get(entry["name"], {})
        original_score = old.get("flightready_score")
        row = {"variant":entry["name"], "family":entry["family"], "prevalence_percent":entry["prevalence"],
               "original_score":original_score, "improved_score":card["measured_quality_score"],
               "original_status":old.get("readiness_status"), "improved_status":card["readiness_decision"],
               "original_coverage_percent":old.get("evidence_coverage_percent"), "improved_coverage_percent":100*card["evidence_coverage"],
               "reference_free_status":free["readiness_decision"], "reference_free_coverage_percent":100*free["evidence_coverage"],
               "mAP50_95":old.get("mAP50_95"), "images":image_count, "expected_image_loss":expected_loss,
               **{k:ann[k] for k in ("box_omission_fraction","class_disagreement_fraction","geometry_disagreement_fraction")},
               "class_js_divergence":class_js, **image_sample}
        row.update(duplicate_fraction=duplicate_fraction)
        rows.append(row); details[entry["name"]] = {"reference_assisted":card,"reference_free":free,"annotation":ann,"image_sample":image_sample,"duplicates":duplicate_evidence}
        integrity.append({"variant":entry["name"],"manifest_images":entry["images"],"observed_images":image_count,"valid":entry["images"]==image_count,
                          "label_files":len(records),"invalid_annotation_rows":invalid})
    return pd.DataFrame(rows), details, integrity


def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n":len(x),"pearson_r":None,"pearson_p":None,"spearman_rho":None,"spearman_p":None}
    pr, pp = pearsonr(x,y); sr, sp = spearmanr(x,y)
    return {"n":len(x),"pearson_r":float(pr),"pearson_p":float(pp),"spearman_rho":float(sr),"spearman_p":float(sp)}


def plots(tab, tab_perf, vis):
    PLOTS.mkdir(parents=True, exist_ok=True)
    def save(name):
        plt.tight_layout(); plt.savefig(PLOTS/name, dpi=180); plt.close()
    means = tab_perf.groupby("variant")["r2"].mean()
    joined = tab.set_index("variant").join(means.rename("performance"))
    plt.scatter(joined.original_score, joined.performance); plt.xlabel("v1 score"); plt.ylabel("Mean R2"); plt.title("Tabular v1 score vs performance"); save("01_original_score_vs_performance.png")
    plt.scatter(joined.improved_score, joined.performance); plt.xlabel("v2 measured quality score"); plt.ylabel("Mean R2"); plt.title("Tabular v2 score vs historical performance"); save("02_improved_score_vs_performance.png")
    for family, group in tab.groupby("family"):
        if group.corruption_percent.nunique()>1: plt.plot(group.corruption_percent, group.improved_score, marker="o", label=family)
    plt.xlabel("Corruption level (%)"); plt.ylabel("v2 score"); plt.legend(); plt.title("Metric response by corruption level"); save("03_corruption_response.png")
    independent = vis[vis.original_score.isna()]
    sensitivity = [float(independent.loc[independent.family==f,"improved_score"].mean() < vis.loc[vis.variant=="clean","improved_score"].iloc[0]) for f in independent.family]
    plt.bar(independent.family, sensitivity); plt.xticks(rotation=35,ha="right"); plt.ylabel("Detected score change"); plt.ylim(0,1.1); plt.title("Held-out controlled-defect sensitivity"); save("04_detection_sensitivity.png")
    x=np.arange(len(tab)); plt.bar(x-.2,tab.original_coverage_percent,width=.4,label="v1"); plt.bar(x+.2,tab.improved_coverage_percent,width=.4,label="v2 assisted"); plt.ylabel("Evidence coverage (%)"); plt.legend(); plt.title("Evidence coverage before and after"); save("05_evidence_coverage.png")
    measured=vis[vis.mAP50_95.notna()]; plt.bar(measured.variant,measured.mAP50_95); plt.xticks(rotation=35,ha="right"); plt.ylabel("mAP50-95"); plt.title("Preserved YOLO11n model measurements"); save("06_model_performance.png")


def root_cause_document():
    text = """# FlightReady Root-Cause Analysis

## Scope and evidence

This analysis preserves the original tabular and VisDrone results as historical evidence. FlightReady had two different outputs: a requirement-based categorical decision and an optional experimental weighted score. They must not be interpreted as the same mechanism.

## Confirmed causes

| Observed behavior | Confirmed cause | Classification | v2 correction |
|---|---|---|---|
| Tabular score stayed 100 with 50% coverage | The experimental score renormalized over only measured components; the two measured components passed while unknown components contributed no penalty. | Misleading aggregation/reporting, not an arithmetic error | Retain unknown weight, expose bounds, and withhold a primary score unless coverage is complete. |
| Missing data produced NOT_READY while score stayed 100 | The decision correctly used confirmed per-feature missingness requirements, but the experimental score did not include those requirement findings. | Inconsistent output scopes | Show gates and score components together; gates determine the decision. |
| Noise and label corruption were missed | Values remained schema-valid. No trusted row/label reference or semantic relationship criterion was supplied. | Missing capability/evidence | Add reference-assisted disagreement checks; without a trusted reference report insufficient evidence. |
| VisDrone incomplete data scored higher | Removing difficult images changed clarity and uniqueness summaries favorably; no expected-image manifest component existed in the score. | Construct-validity/coverage gap | Add expected acquisition count/segment evidence; otherwise completeness is unavailable. |
| Wrong classes did not change the score | Syntax, class presence and distributions cannot establish per-object semantic correctness. | Missing semantic evidence, not a direction bug | Support trusted annotation audits and explicitly label reference-assisted agreement. |
| Combined defects had the highest v1 score | Omitted annotations reduced the population of hard/small objects and image removal/noise shifted proxy distributions; compensatory weighted means rewarded easier remaining evidence. | Aggregation and proxy-confounding problem | Use non-compensatory gates, separate dimensions, reference comparisons, and metric-level contributions. |
| Every VisDrone variant was NOT_READY | A confirmed hard requirement failed across the common dataset; the score was never the decision. | Requirement outcome; not a score bug | Preserve gate trace and explain the common failure independently of quality summaries. |
| Image noise barely affected clarity | Laplacian variance is increased by both detail and noise, so it is not a monotonic noise measure. | Method limitation | Report multiple proxies and paired/reference evidence; do not claim any single proxy proves corruption. |

## Methodology boundary

V2 does not infer semantic label correctness from class histograms, or acquisition completeness from the observed file count alone. Those dimensions remain unavailable without a trusted audit or expected-acquisition manifest. Reference-assisted results in the controlled experiments therefore use more information than the original reference-free run and are labelled accordingly.
"""
    (DOCS/"FlightReady_Root_Cause_Analysis.md").write_text(text,encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True); PLOTS.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    tab, tab_perf, tab_details, tab_integrity = tabular_validation()
    vis, vis_details, vis_integrity = visdrone_validation()
    tab.to_csv(OUT/"tabular_comparison.csv",index=False); vis.to_csv(OUT/"visdrone_comparison.csv",index=False)
    (OUT/"metric_details.json").write_text(json.dumps({"tabular":tab_details,"visdrone":vis_details},indent=2,allow_nan=False),encoding="utf-8")
    (OUT/"integrity_verification.json").write_text(json.dumps({"tabular":tab_integrity,"visdrone":vis_integrity},indent=2),encoding="utf-8")
    plots(tab,tab_perf,vis)
    tab_mean = tab_perf.groupby("variant")["r2"].mean(); tab_join=tab.set_index("variant").join(tab_mean.rename("r2"))
    vis_measured=vis[vis.mAP50_95.notna()]
    stats={"tabular_v1":corr(tab_join.original_score.to_numpy(float),tab_join.r2.to_numpy(float)),
           "tabular_v2":corr(tab_join.improved_score.to_numpy(float),tab_join.r2.to_numpy(float)),
           "visdrone_v1":corr(vis_measured.original_score.to_numpy(float),vis_measured.mAP50_95.to_numpy(float)),
           "visdrone_v2":corr(vis_measured.improved_score.to_numpy(float),vis_measured.mAP50_95.to_numpy(float)),
           "interpretation":"Descriptive only; variants are related interventions and the VisDrone sample is n=7."}
    (OUT/"statistical_analysis.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    root_cause_document()
    summary={"tabular_variants":len(tab),"visdrone_variants":len(vis),"historical_visdrone_models":len(vis_measured),
             "independent_assessment_variants":int(vis.original_score.isna().sum()),"integrity_failures":sum(not x["valid"] for x in tab_integrity+vis_integrity),"statistics":stats}
    (OUT/"validation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
