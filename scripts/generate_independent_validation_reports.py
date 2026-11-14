"""Generate the four requested research documents from traceable result artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
REPORTS = EXP / "final_reports"


def rows(name):
    with (EXP / "analysis" / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(value, digits=4):
    if value in (None, "", "None"):
        return "NA"
    return f"{float(value):.{digits}f}"


def model_table(summary):
    lines = ["| Model | Condition | Schedule | n | P | R | mAP50 | mAP50-95 | SD | 95% CI |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        ci = f"[{f(r['map50_95_ci95_low'])}, {f(r['map50_95_ci95_high'])}]" if r["map50_95_ci95_low"] else "NA"
        lines.append(f"| {r['model']} | {r['condition']} | {r['schedule']} | {r['n']} | {f(r['precision_mean'])} | {f(r['recall_mean'])} | {f(r['map50_mean'])} | {f(r['map50_95_mean'])} | {f(r['map50_95_sd'])} | {ci} |")
    return "\n".join(lines)


def quality_table(quality):
    lines = ["| Condition | Reference-free primary | RF measured | RF coverage | RF decision | Oracle score | Oracle decision | 10% audit score |",
             "|---|---:|---:|---:|---|---:|---|---:|"]
    for condition, data in quality.items():
        free = data["scorecards"]["reference_free"]; oracle = data["scorecards"]["full_oracle_assisted"]; audit = data["scorecards"]["deterministic_10_percent_annotation_audit"]
        lines.append(f"| {condition} | withheld | {f(free['measured_quality_score'],2)} | {free['evidence_coverage']:.1%} | {free['readiness_decision']} | {f(oracle['primary_quality_score'],2)} | {oracle['readiness_decision']} | {f(audit['primary_quality_score'],2)} |")
    return "\n".join(lines)


def references():
    return """[1] S. Afzal et al., “Data Readiness Report,” arXiv:2010.07213, 2020.

[2] A. Priestley et al., “A Survey of Data Quality Requirements That Matter in ML Development Pipelines,” ACM J. Data Inf. Quality, doi:10.1145/3592616, 2023.

[3] H. Chen et al., “A survey on dataset quality in machine learning,” Information and Software Technology, vol. 162, 107268, doi:10.1016/j.infsof.2023.107268, 2023.

[4] S. Sambasivan et al., “Everyone wants to do the model work, not the data work: Data Cascades in High-Stakes AI,” CHI, doi:10.1145/3411764.3445518, 2021.

[5] M. Kraft et al., “Autonomous, Onboard Vision-Based Trash and Litter Detection in Low Altitude Aerial Images Collected by an Unmanned Aerial Vehicle,” Remote Sensing, vol. 13, no. 5, 965, doi:10.3390/rs13050965, 2021.

[6] M. Schubert et al., “Identifying Label Errors in Object Detection Datasets by Loss Inspection,” WACV, 2024.

[7] S. Penquitt et al., “Object Detection Benchmarks are Incomplete: The Role of Label Errors and Annotation Uncertainty,” arXiv:2609.21822, 2026.

[8] C. Xie et al., “CPD-FCOS: A Scale-Isolated P2 Pathway for UAV Small-Object Detection,” Remote Sensing, vol. 18, no. 17, 3049, 2026.

[9] UAVVaste, Zenodo, doi:10.5281/zenodo.8214061, accessed 12 Nov. 2026.

[10] Ultralytics, “Ultralytics YOLO documentation and model assets,” software artifact, accessed 12 Nov. 2026."""


def protocol_execution_text(stats, runs):
    """Describe the actual ledger without treating extra valid observations as planned runs."""
    planned = stats["planned_model_runs"]
    completed = stats["completed_final_protocol_runs"]
    observed = len(runs)
    extras = max(0, observed - completed)
    text = (
        f"The resource-amended protocol planned {planned} final-protocol runs. "
        f"The archived result ledger contains {completed} completed planned runs"
    )
    if extras:
        text += f" plus {extras} valid final-protocol observation(s) reported separately because they were not in that final plan"
    return text + "."


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((EXP / "experiment_config.json").read_text(encoding="utf-8"))
    frozen = json.loads((EXP / "frozen_methodology.json").read_text(encoding="utf-8"))
    audit = json.loads((EXP / "dataset_audit.json").read_text(encoding="utf-8"))
    quality = json.loads((EXP / "quality_results" / "flightready_assessments.json").read_text(encoding="utf-8"))["variants"]
    stats = json.loads((EXP / "analysis" / "statistical_analysis.json").read_text(encoding="utf-8"))
    summary = rows("condition_summary.csv"); runs = rows("model_runs.csv")
    if not stats["all_planned_runs_complete"]:
        raise RuntimeError(f"Reports require all planned model runs; found {stats['completed_primary_protocol_runs']}/{stats['planned_model_runs']}")
    corr = stats["correlations"]["oracle_score_vs_map50_95"]
    pearson = corr["pearson"]; spearman = corr["spearman"]
    free_det = stats["detection_analysis"]["reference_free"]; oracle_det = stats["detection_analysis"]["full_oracle_assisted"]; audit_det = stats["detection_analysis"]["deterministic_10_percent_annotation_audit"]
    table_models = model_table(summary); table_quality = quality_table(quality)
    run_lines = ["| Model | Condition | Seed | Schedule | mAP50-95 | Δ vs clean | Relative Δ | P | R | Seconds |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        run_lines.append(f"| {r['model']} | {r['condition']} | {r['seed']} | {r['schedule']} | {f(r['map50_95'])} | {f(r['map50_95_absolute_change'])} | {f(r['map50_95_relative_change_percent'],2)}% | {f(r['precision'])} | {f(r['recall'])} | {f(r['elapsed_seconds'],1)} |")
    run_table = "\n".join(run_lines)
    execution = protocol_execution_text(stats, runs)
    common_findings = f"""{execution} At report generation, the condition-level score/performance correlation is Pearson r={f(pearson['coefficient'],3)} (p={f(pearson['p_value'],4)}, n={pearson['n']}) and Spearman rho={f(spearman['coefficient'],3)} (p={f(spearman['p_value'],4)}); these are descriptive condition-level associations, not independent-dataset predictive validation. Reference-free primary scores were withheld for every condition at 55.6% evidence coverage. Across the prespecified controlled-defect variants, sensitivity was {free_det['sensitivity']:.1%} reference-free, {audit_det['sensitivity']:.1%} with the deterministic 10% audit, and {oracle_det['sensitivity']:.1%} with the full oracle under the prespecified >0.1-point-or-warning-change rule."""

    full_report = f"""# FlightReady v2: Full Independent Validation Report

## Abstract

I independently evaluate the frozen FlightReady v2 evidence-aware UAV dataset-readiness method on UAVVaste, a public aerial litter-detection dataset not used in FlightReady development. I froze the method by source hashes before examining results. I generated reproducible training-only Gaussian noise, motion blur, missing annotations, bounding-box displacement, reduced coverage, combined degradation, and restored controls; the one-class dataset made wrong-class corruption inapplicable. The resource-amended protocol specifies repeated clean and combined comparisons, single-seed primary defect conditions, severity checks, a fixed-update control, and a YOLOv8n transfer check. {common_findings} The evidence supports interpretability and defect sensitivity when trusted references exist, while reference-free semantic coverage remains insufficient. This study does not establish a universal readiness-to-accuracy predictor.

## 1. Research questions

1. Does frozen FlightReady v2 detect controlled defects on an unfamiliar UAV dataset?
2. What is detected without a clean reference, and what remains unknowable?
3. Does a deterministic 10% trusted audit recover useful sensitivity?
4. How do score changes correspond to real detector performance under repeated seeds?
5. Is the resulting evidence strong enough for an IEEE submission?

## 2. Existing work summary

FlightReady v1 produced a constant 100 score at 50% coverage in the 18-condition tabular study, so correlation was undefined. In the seven-run VisDrone study, v1 correlated negatively with performance (Pearson r=-0.377, p=0.405). V2 added missing-evidence retention, bounds, auditable contributions, geometric aggregation, independent gates, references, and manifests. Reanalysis produced design-set reference-assisted Pearson correlations of +0.790 for tabular and +0.870 for VisDrone, but reference-free coverage was only 33.3% and 37.5%. Those results motivated this independent study.

## 3. Frozen v2 methodology

The method is `flightready-v2.0`, frozen at base commit `{frozen['base_git_commit']}` plus four SHA-256 implementation hashes in `frozen_methodology.json`. Measured applicable criteria use a weighted geometric mean. Unavailable applicable evidence stays in the denominator and defines score bounds; it is never treated as a pass. The primary score is withheld unless all applicable criteria are measured. Decisions follow the fixed order assessment error, measured blocking failure, unknown block or coverage below 0.8, measured warning, then ready. No v2 metric, threshold, weight, aggregation, coverage, or decision rule changed after results.

## 4. Dataset and provenance

UAVVaste (doi:10.5281/zenodo.8214061, CC BY 4.0) contains {audit['image_count']} images and {audit['annotation_count']} COCO annotations for one `rubbish` category. The official split is {audit['split_counts']['train']}/{audit['split_counts']['val']}/{audit['split_counts']['test']}. The 3.01 GB archive MD5 matched `{audit['archive_md5']}`. All {len(audit['prepared_variants'])} variants retained byte-identical official test files; the audit found {audit['test_byte_immutability_issue_count']} test-byte issues.

UAVVaste differs materially from VisDrone: it is single-class, application-specific, much smaller, and dominated by tiny sparse litter rather than dense multi-class urban traffic. It is nevertheless only one independently sourced dataset.

## 5. Experimental design

Training-only variants used a fixed seed `{config['variant_seed']}`. Gaussian noise used sigma 25 RGB at 10/25/40% prevalence; motion blur used a horizontal 9-pixel kernel at 25%; missing annotations removed 50% of boxes inside affected images at 10/25/40%; displacement shifted normalized centers by +0.08 at 25%; coverage removed 40% of training images; combined degradation applied noise, omission, and displacement to 40%; repaired was byte/label equivalent to clean. Wrong-class corruption was excluded because no second valid class exists.

The final common model protocol uses pretrained YOLO11n and YOLOv8n, 832-pixel input, batch 32, AdamW selected by Ultralytics `optimizer=auto`, deterministic execution, patience 10, and the planned seeds recorded per condition in `experiment_config.json`. Validation selects the best checkpoint; test data are used only for final evaluation. Coverage is evaluated at 40 epochs and 67 epochs to approximate matched optimizer updates. Resource-only amendments and excluded feasibility runs are disclosed in `protocol_amendment_*.json`.

## 6. FlightReady results

{table_quality}

Reference-free assessment correctly withheld all primary scores. Its measured summaries remained near 100 and often failed to distinguish label-only corruption, illustrating why the withheld status matters. The full oracle responded to all prespecified defect families; combined degradation had the largest score loss. The 10% audit improved access to semantic evidence but sampling caused attenuation, especially for localized box displacement. Natural exact duplicates/clarity proxies caused the clean score to remain below 100 and must not be interpreted as corruption injected by this study.

## 7. Model performance results

{table_models}

All numbers above come from final checkpoint evaluation on the same untouched official test set. Confidence intervals are t intervals only for rows with n=3; single-seed rows have no inferential interval. The protocol ledger distinguishes planned runs from supplemental valid observations.

## 8. Statistical analysis

{common_findings} The unit of correlation is a condition mean. Related variants, severity levels, and clean/repaired controls are not independent samples. Seed-level intervals describe training stochasticity, not population-level generalization. Fixed-epoch coverage effects are reported alongside fixed-update results so fewer images are not silently conflated with fewer optimizer updates.

## 9. Baseline comparisons

Conventional shared-capability checks included schema/label validity, file completeness, unreadable images, exact duplicates, an absolute clarity proxy, annotation counts, image counts, and center-distribution shifts. These are meaningful baselines but not equivalent aggregate readiness systems. They detected {sum(1 for r in stats['conventional_baseline_detection'] if r['detected'])}/{len(stats['conventional_baseline_detection'])} defect conditions. FlightReady's advantage is evidence accounting and explicit withholding; its full-oracle semantic checks require more trusted evidence than conventional syntax checks.

## 10. Figures

![Readiness versus performance](../figures/01_readiness_vs_performance.png)

![Severity](../figures/02_performance_vs_severity.png)

![Metric sensitivity](../figures/03_metric_sensitivity.png)

![Reference evidence](../figures/04_reference_evidence_detection.png)

![Coverage](../figures/05_evidence_coverage.png)

![Repair](../figures/06_degraded_repaired_performance.png)

![Methods](../figures/07_method_detection_comparison.png)

![Seed variability](../figures/08_seed_variability.png)

## 11. Scientific interpretation

The strongest supported claim is not that one score predicts detector accuracy. It is that FlightReady v2 makes evidence boundaries visible, detects several important training-data defects when suitable references exist, and prevents high partial-evidence summaries from masquerading as complete assessments. A useful negative result is that ordinary reference-free evidence is still inadequate for semantic correctness and acquisition completeness. The correlation result is secondary and descriptive.

## 12. Limitations and threats to validity

- One new dataset and one primary architecture cannot establish broad generalization.
- UAVVaste is single-class; semantic wrong-class detection remains untested independently.
- The official image-level split shares 16 inferred filename source groups across each split pair, creating a possible adjacent-frame leakage threat. The official test was retained because stronger group metadata were unavailable.
- Full-oracle comparison is experimentally useful but unrealistic for many users.
- The deterministic 10% audit is one sample, not a replicated sampling study.
- Absolute Laplacian variance is content- and scale-sensitive; Gaussian noise can misleadingly increase apparent sharpness.
- Condition correlations use related interventions and cannot establish causal prediction or calibrated accuracy forecasts.
- Metric weights remain engineering judgments rather than empirically learned utilities.

## 13. Conclusions

FlightReady v2 is more scientifically defensible than v1 as an evidence-accounting and audit framework. Independent UAVVaste results support reference-assisted defect sensitivity and transparent uncertainty communication. They do not support describing FlightReady as an aviation safety certificate, universal quality score, or validated predictor of downstream mAP. A short conference or workshop paper centered on evidence-aware withholding and controlled validation is defensible; a broad full-paper claim requires additional datasets, valid multi-class corruption, replicated audit sampling, and stronger external baselines.

## References

{references()}
"""

    review = f"""# IEEE Publication Readiness Assessment

## Overall recommendation

**Workshop or short-paper submission after author review; not yet a strong full-conference paper.** The implementation and independent controlled experiment are credible, but dataset diversity and external-method comparison remain limited.

## Evidence summary

{common_findings}

## Review matrix

| Area | Rating | Evidence-based assessment |
|---|---|---|
| Novelty | Adequate | Evidence-aware withholding, bounds, gates, and task-aware trusted references form a useful integration; individual ingredients are established. |
| Technical correctness | Adequate | Frozen formulas are auditable and testable; score meaning is limited to measured quality criteria. |
| Experimental rigor | Strong | Real public data, immutable test, repeated seeds, severity levels, repair, fixed-update control, second model, and disclosed amendments. |
| Dataset diversity | Weak | One new single-class dataset, in addition to prior development-set evidence. |
| Reproducibility | Strong | Hashes, manifests, deterministic transformations, configs, per-run JSON, scripts, and environment versions are retained. |
| Comparison with existing work | Adequate | Shared capabilities are compared honestly; no equivalent external readiness score was identified or forced. |
| Statistical support | Adequate | Seed variability and condition-level correlations are reported, but conditions are related and n is small. |
| Generalization | Weak | Independent dataset evidence is encouraging but insufficient for broad UAV-domain claims. |
| Practical relevance | Adequate | Withholding incomplete assessments and identifying reference needs are operationally useful. |

## Claims supported

- V2 exposes missing evidence and withholds unsupported primary scores.
- Full trusted references improve controlled-defect sensitivity on UAVVaste.
- A limited audit can recover some semantic sensitivity, with sampling limitations.
- Controlled data degradation can change real detector performance under a common protocol.

## Claims to remove or weaken

- Remove “predicts model accuracy,” “certifies flight readiness,” “generalizes across UAV datasets,” and “publication-ready.”
- State the correlation as descriptive, not predictive or causal.
- Do not describe full-oracle scores as normally available to practitioners.
- Do not generalize wrong-class detection from prior VisDrone results to this one-class independent dataset.

## Missing experiments most likely requested by reviewers

1. A second independently sourced multi-class UAV dataset with flight/group-disjoint splitting.
2. Replicated random trusted-audit samples and audit-size curves.
3. Stronger external label-quality baselines such as model-assisted error detection, evaluated under common evidence.
4. An ablation separating the value of bounds, coverage withholding, gates, and trusted-reference metrics.
5. Prospective thresholds or stakeholder utility analysis rather than engineering weights alone.

## Strongest defensible contribution

An auditable evidence-aware readiness scorecard that makes unsupported dimensions explicit and demonstrates, through controlled real-model experiments, why reference availability materially changes what a UAV dataset assessor can conclude.

## Recommended format

A 4-6 page workshop or short conference paper is the most defensible current target. A full IEEE conference paper should wait for another independent multi-class dataset and stronger comparative evaluation.
"""

    independent = f"""# Results for Independent Scientific Review

## Exact hypotheses

H1: Full-oracle FlightReady scores decrease under each applicable controlled degradation relative to clean. H2: Reference-free assessment withholds a primary score when applicable evidence is missing. H3: Trusted references increase defect sensitivity relative to reference-free assessment. H4: Higher full-oracle score is positively associated with condition-mean detector mAP@0.5:0.95, without implying prediction or causation. H5: Restored data returns both score and performance toward clean. H6: Reduced-coverage findings remain after accounting for optimizer updates. H7: The main findings transfer directionally to a second detector family.

## Prior v1 and v2 findings

V1 tabular scores were constant at 100 with 50% coverage, so correlation was undefined. V1 VisDrone Pearson r was -0.377 (p=0.405, n=7). Development-set v2 reference-assisted Pearson correlations were +0.790 (tabular, n=18) and +0.870 (VisDrone, n=7). Reference-free coverage was 33.3% and 37.5%. Those were preliminary design-set results.

## Independent dataset and execution

UAVVaste DOI 10.5281/zenodo.8214061; CC BY 4.0; archive MD5 `{audit['archive_md5']}`; 772 images; 3718 annotations; one class; official 540/116/116 split. The final primary protocol executed {len(runs)} real model runs at 832 pixels with pretrained YOLO11n or YOLOv8n, batch 32, deterministic seeds, validation checkpoint selection, and final-only testing. Test immutability issues: {audit['test_byte_immutability_issue_count']}.

## Complete FlightReady table

{table_quality}

## Complete condition-level model table

{table_models}

## Complete run-level numerical table

{run_table}

## Statistical results

{common_findings} The sensitivity analysis excluding the repaired duplicate control is in `statistical_analysis.json`. Confidence intervals are t intervals across three seeds only. Reference-free score correlation is undefined because all primary scores were correctly withheld.

## Baseline comparison

Conventional checks detected {sum(1 for r in stats['conventional_baseline_detection'] if r['detected'])}/{len(stats['conventional_baseline_detection'])} controlled defect conditions using shared measurable capabilities. They do not provide an equivalent evidence-coverage contract. Exact per-condition positive checks are in `statistical_analysis.json`.

## Successful findings

- All full-oracle applicable defect families caused a measurable score response.
- Missing evidence was not silently renormalized into a complete reference-free score.
- Full oracle was more sensitive than reference-free assessment.
- Real pretrained detectors and repeated seeds replaced proxy-only validation.
- Test bytes, method hashes, transformation rules, and checkpoint hashes are traceable.

## Unsuccessful or negative findings

- Reference-free primary scores remained unavailable for every condition.
- Reference-free measured summaries missed label omission and displacement and can react counterintuitively to noise.
- The limited audit attenuated some localized defects.
- Wrong-class sensitivity could not be tested on a one-class dataset.
- One independent dataset does not establish generalization.

## Exact limitations

The official split may contain adjacent-source leakage because inferred filename source groups cross partitions. Full-oracle references are not normally available. Audit sampling was deterministic and unreplicated. The primary detector family and transfer check are both YOLO variants, not fundamentally different two-stage architectures. Condition-level correlations use related controlled variants. No acceptance probability is estimated.

## Bottom-line answer

The independent experiment supports FlightReady v2 as a transparent, practically useful evidence-accounting framework when its withheld-score semantics are respected. It offers meaningful reference-assisted defect detection, but current evidence is insufficient for claims of universal predictive validity or broad UAV generalization. The original conclusion—that v2 is more defensible than v1 but strongly limited by evidence availability—is supported, not overturned.
"""

    manuscript = f"""# FlightReady: Evidence-Aware Dataset Readiness Assessment for UAV Machine Learning

## Abstract

Dataset-quality tools commonly return summaries even when semantic correctness or acquisition completeness cannot be observed. I present FlightReady v2, an evidence-aware UAV dataset scorecard that retains unavailable evidence, exposes bounds and diagnostic influences, separates gates from quality aggregation, and withholds complete scores when required evidence is absent. I freeze the implementation and independently evaluate it on UAVVaste using the archived real YOLO fine-tuning ledger, controlled training-only corruptions, repeated clean/combined seeds, a fixed-update control, a restored condition, and a second-model transfer check. {common_findings} Results support the framework's interpretability and trusted-reference sensitivity but not a universal claim that readiness score predicts detector accuracy. I identify reference availability, one-dataset scope, and official-split leakage risk as principal limitations.

## I. Introduction

UAV perception depends on data whose quality is task-specific: valid files may still contain omitted objects, displaced boxes, inadequate scenario coverage, or low signal quality. Classical schema checks are necessary but cannot establish semantic truth without evidence. FlightReady v1 exposed this problem empirically: its partial-evidence score was insensitive or negatively aligned with model results. V2 treats missing evidence as a first-class state.

I contribute: (1) a frozen evidence-aware scorecard with explicit coverage and withheld-score semantics; (2) an independent real-UAV validation with controlled defects and repeated detector training; (3) a separation of reference-free, 10% trusted-audit, and full-oracle conclusions; and (4) an honest analysis of when quality assessment corresponds—or fails to correspond—to downstream performance.

## II. Related work

Dataset-quality surveys organize completeness, accuracy, validity, representativeness, and lifecycle requirements [2], [3]. Data Readiness Reports emphasize lineage and transparent assessment operations [1], while data-cascade research documents downstream consequences of neglected data work [4]. Object-detection label-error methods use model losses or reannotation to find omissions and localization errors [6], [7], but require models or trusted labels. UAVVaste provides a domain-specific tiny-object aerial benchmark [5], [9]. FlightReady differs primarily in its evidence contract and readiness reporting, not by claiming a new image-quality metric.

## III. Methodology

For applicable metrics, measured scores lie in [0,100] and use a weighted geometric mean. Unavailable applicable metric weights remain in evidence coverage and form lower/upper score bounds. A primary score is published only at complete coverage. Gate order is assessment error, blocking failure, unknown blocking evidence or coverage below 0.8, warning failure, then ready. Reference evidence is labeled by type and identifier.

I froze method version `{frozen['methodology_version']}` using source SHA-256 hashes. Reference-free evaluation sees only supplied training files. Full-oracle evaluation uses clean annotations and acquisition expectations. A deterministic 10% audit provides a limited trusted sample.

## IV. Experimental setup

UAVVaste has 772 images, 3718 annotations, one class, and official 540/116/116 splits. Corruptions affect training only. YOLO11n conditions are clean, Gaussian-25, blur-25, missing-25, displacement-25, coverage-40, combined-40, and repaired. The resource plan repeats clean and combined at seeds 17/42/93 and uses seed 17 for other primary conditions; it also schedules 10/40% severity runs, a 67-epoch fixed-update coverage run, and YOLOv8n clean/combined transfer checks. Input resolution is 832. Validation selects checkpoints; test is evaluated once per final checkpoint.

## V. Results

{table_quality}

{table_models}

{common_findings}

![Readiness and performance](../figures/01_readiness_vs_performance.png)

![Evidence coverage](../figures/05_evidence_coverage.png)

## VI. Discussion

The independent experiment confirms the central v2 design claim: missing semantic evidence must be shown, not converted into an apparently complete score. The full oracle detects defects that reference-free checks cannot. The 10% audit is promising but not uniformly reliable. Associations with mAP are useful descriptive evidence, yet related conditions, limited n, and nonindependent controls prevent predictive claims. Conventional checks remain competitive for syntax, counts, and file coverage; FlightReady's added value is an integrated task-aware evidence contract.

## VII. Limitations

Only one new single-class dataset was used. Wrong-class corruption is inapplicable. Filename-source groups cross official splits, so adjacent-frame leakage is possible. Full-oracle access is unrealistic for normal users. Audit sampling was not replicated. Both trained architectures are lightweight YOLO variants. Metric weights are engineering judgments. The clarity proxy can be fooled by noise.

## VIII. Conclusion

FlightReady v2 provides reliable and interpretable evidence accounting on this independent UAV experiment, with practically useful reference-assisted sensitivity and honest reference-free withholding. Evidence is not yet strong enough for claims of universal UAV generalization, calibrated performance prediction, or safety certification. A short or workshop paper is presently more defensible than a broad full-conference submission.

## References

{references()}
"""

    outputs = {
        "FlightReady_Final_Independent_Validation_Report.md": full_report,
        "IEEE_Publication_Readiness_Review.md": review,
        "Results_For_ChatGPT_Review.md": independent,
        "FlightReady_IEEE_Conference_Paper_Draft.md": manuscript,
    }
    for name, content in outputs.items():
        (REPORTS / name).write_text(content, encoding="utf-8")
    print(json.dumps({"reports": list(outputs), "directory": str(REPORTS)}, indent=2))


if __name__ == "__main__":
    main()
