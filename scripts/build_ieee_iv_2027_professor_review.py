"""Build the audited IEEE IV 2027 FlightReady manuscript package.

The manuscript is generated only from frozen FlightReady v2 artifacts and the
completed independent-validation report. This script does not run experiments
or modify methodology files.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
OUT = ROOT / "submission"
FIG = EXP / "figures"
STEM = "FlightReady_IEEE_IV_2027_FINAL"

# Prefer Times New Roman on Windows and a metrically compatible serif family
# elsewhere. All four faces are embedded rather than relying on PDF base fonts.
font_sets = [
    (Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "times.ttf",
     Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "timesbd.ttf",
     Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "timesi.ttf",
     Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "timesbi.ttf"),
    (Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
     Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
     Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf"),
     Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf")),
]
font_files = next((faces for faces in font_sets if all(path.exists() for path in faces)), None)
if font_files is None:
    raise FileNotFoundError("Times New Roman or Liberation Serif font family is required to build the manuscript PDF")
pdfmetrics.registerFont(TTFont("TNR", str(font_files[0])))
pdfmetrics.registerFont(TTFont("TNR-Bold", str(font_files[1])))
pdfmetrics.registerFont(TTFont("TNR-Italic", str(font_files[2])))
pdfmetrics.registerFont(TTFont("TNR-BoldItalic", str(font_files[3])))
pdfmetrics.registerFontFamily(
    "TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic"
)


def make_equation_figure():
    """Render the corrected aggregation equation consistently in DOCX and PDF."""
    target = OUT / "figures" / "weighted_geometric_mean_equation.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.5, 0.78), dpi=260)
    fig.patch.set_alpha(0)
    fig.text(
        0.5, 0.69,
        r"$W_A=\sum_{i\in A}w_i,\quad W_M=\sum_{i\in M}w_i,\quad c=\frac{W_M}{W_A}$",
        ha="center", va="center", fontsize=12,
    )
    fig.text(
        0.5, 0.24,
        r"$Q_M=100\exp\!\left(\frac{1}{W_M}\sum_{i\in M}w_i\ln\!\left(\frac{q_i}{100}\right)\right).\qquad (1)$",
        ha="center", va="center", fontsize=12,
    )
    fig.savefig(target, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return target


def stage_latex_figures():
    target_dir = OUT / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "01_readiness_vs_performance.png",
        "02_performance_vs_severity.png",
        "04_reference_evidence_detection.png",
        "07_method_detection_comparison.png",
        "08_seed_variability.png",
    ):
        shutil.copy2(FIG / name, target_dir / name)


def load_evidence():
    stats = json.loads((EXP / "analysis" / "statistical_analysis.json").read_text(encoding="utf-8"))
    quality = json.loads((EXP / "quality_results" / "flightready_assessments.json").read_text(encoding="utf-8"))["variants"]
    frozen = json.loads((EXP / "frozen_methodology.json").read_text(encoding="utf-8"))
    with (EXP / "analysis" / "condition_summary.csv").open(encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    return stats, quality, frozen, summary


def evidence_model(stats, quality, frozen, summary):
    by = {(r["model"], r["condition"], r["schedule"]): r for r in summary}
    get = lambda condition, model="yolo11n", schedule="fixed_epochs": by[(model, condition, schedule)]
    corr = stats["correlations"]["oracle_score_vs_map50_95"]
    sens = stats["correlations"]["sensitivity_excluding_repaired_duplicate_control"]

    quality_rows = [["Condition", "RF", "Coverage", "Oracle", "Audit", "Decision"]]
    for key, label in [
        ("clean", "Clean"), ("gaussian_25", "Gaussian 25%"),
        ("motion_blur_25", "Blur 25%"), ("missing_annotations_25", "Missing 25%"),
        ("bbox_displacement_25", "Box shift 25%"), ("coverage_40", "Coverage -40%"),
        ("combined_40", "Combined 40%"), ("repaired", "Repaired"),
    ]:
        q = quality[key]["scorecards"]
        rf = q["reference_free"]
        oracle = q["full_oracle_assisted"]
        audit = q["deterministic_10_percent_annotation_audit"]
        quality_rows.append([
            label,
            "withheld",
            f"{100 * float(rf['evidence_coverage']):.1f}%",
            f"{float(oracle['primary_quality_score']):.2f}",
            f"{float(audit['primary_quality_score']):.2f}",
            {"READY": "Ready", "CONDITIONALLY_READY": "Conditional"}.get(
                oracle["readiness_decision"], oracle["readiness_decision"]
            ),
        ])

    primary_rows = [["Condition", "n", "mAP50-95", "SD", "Relative change"]]
    for key, label in [
        ("clean", "Clean"), ("gaussian_25", "Gaussian 25%*"),
        ("motion_blur_25", "Blur 25%"), ("missing_annotations_25", "Missing 25%"),
        ("bbox_displacement_25", "Box shift 25%"), ("coverage_40", "Coverage -40%"),
        ("combined_40", "Combined 40%"), ("repaired", "Repaired"),
    ]:
        r = get(key)
        change = r["paired_map50_95_relative_change_percent_mean"]
        primary_rows.append([
            label, r["n"], f"{float(r['map50_95_mean']):.4f}",
            "-" if not r["map50_95_sd"] else f"{float(r['map50_95_sd']):.4f}",
            "baseline" if key == "clean" else f"{float(change):.2f}%",
        ])

    extended_rows = [["Comparison", "Reference", "Result", "Change/interpretation"]]
    extended_rows.extend([
        ["YOLO11n combined, 3 seeds", "0.4790", "0.4452", "-7.05%"],
        ["Coverage 40%, fixed updates", "0.4724", "0.4472", "-5.32%"],
        ["YOLOv8n combined, seed 17", "0.4861", "0.4454", "-8.36%"],
        ["Gaussian 10% -> 40%", "0.4682", "0.4621", "severity trend"],
        ["Missing 10% -> 40%", "0.4687", "0.4467", "severity trend"],
    ])

    history_rows = [["Study", "Scope", "Readiness-performance result", "Interpretation"]]
    history_rows.extend([
        ["Tabular v1", "108 models; 18 conditions", "Score constant 100 at 50% coverage", "Correlation undefined"],
        ["VisDrone v1", "7 YOLO11n runs", "r=-0.377, p=.405", "No supported association"],
        ["Tabular v2 reassessment", "Preserved design-set results", "r=.790, p=.000096", "Method-development evidence"],
        ["VisDrone v2 reassessment", "Preserved design-set results", "r=.870, p=.0108", "Method-development evidence"],
        ["UAVVaste v2", "19 planned + 2 supplemental", "r=.767, p=.0263; rho=.838, p=.0093", "Descriptive; not predictive validation"],
    ])

    hashes = "; ".join(f"{Path(k).name} {v[:12]}" for k, v in frozen["frozen_files"].items())
    return {
        "corr": corr,
        "sens": sens,
        "quality_rows": quality_rows,
        "primary_rows": primary_rows,
        "extended_rows": extended_rows,
        "history_rows": history_rows,
        "hashes": hashes,
        "clean_ci": (float(get("clean")["map50_95_ci95_low"]), float(get("clean")["map50_95_ci95_high"])),
    }


TITLE = "FlightReady v2: Evidence-Aware Data Readiness for Intelligent UAV Systems"
AUTHOR = "Anonymous submission"
ROLE = "Double-blind review"


def sections(ev):
    pearson = ev["corr"]["pearson"]
    spearman = ev["corr"]["spearman"]
    p7 = ev["sens"]["pearson"]
    s7 = ev["sens"]["spearman"]
    return [
        ("I. INTRODUCTION", [
            "Machine-learning systems can fail because their training data are incomplete, mislabeled, unrepresentative, or supported by insufficient evidence. Data-quality tools usually report the checks they can compute, but a high partial score can be mistaken for a complete readiness judgment. I developed FlightReady to investigate whether a dataset supplies sufficient quality evidence for a stated intelligent-system task while keeping missing evidence visible.",
            "The project progressed through three empirical stages. A tabular study first exposed a failure mode in FlightReady v1: the score remained 100 at 50% evidence coverage while regression performance declined. A VisDrone object-detection study then produced a negative, nonsignificant v1 association. I used those design-set results to formulate and freeze FlightReady v2, which separates quality aggregation, evidence coverage, and readiness gates. I subsequently evaluated the frozen method on UAVVaste, an independently sourced UAV dataset not used to design v2.",
            "The research questions are: (RQ1) Does frozen v2 respond to controlled defects in an unfamiliar UAV dataset? (RQ2) Which defects are detectable without trusted reference evidence, with a limited audit, and with a full oracle? (RQ3) How do readiness scores correspond to detector performance under the completed protocol? (RQ4) Which conclusions remain unsupported?",
            "This paper contributes a context-aware evidence model that withholds the primary score when applicable evidence is incomplete; an implementable, frozen aggregation and decision procedure; controlled evidence spanning tabular, VisDrone, and UAVVaste studies; and an explicit negative result that reference-free checks remain inadequate for semantic correctness and acquisition completeness. I do not claim that FlightReady predicts detector accuracy, establishes causal effects, or certifies operational safety.",
        ]),
        ("II. RELATED WORK", [
            "Data-readiness work treats quality as contextual and connected to downstream use rather than as a context-free dataset property [1]-[3]. Data-cascade research shows that collection and labeling problems propagate through model development [4], while datasheets encourage explicit documentation of composition, provenance, collection, and recommended use [5]. FlightReady complements documentation by producing a task-specific assessment record with metric provenance, unavailable-evidence states, weighted coverage, score bounds, and independent decision gates.",
            "These mechanisms differ from conventional data validation. Schema validity, file readability, duplicates, annotation counts, and simple distribution comparisons are useful baselines, but they cannot establish whether labels are semantically correct or whether acquisition is complete without trusted evidence. FlightReady therefore represents the boundary between measured quality and unknown quality rather than converting unknown evidence into a pass.",
            "VisDrone is a large multi-class UAV object-detection benchmark with small objects, occlusion, viewpoint variation, and clutter [6]. UAVVaste targets low-altitude aerial litter detection and contains sparse, small instances in a one-class setting [7], [8]. Its domain and label structure differ from the urban imagery used during the VisDrone design study, making it suitable for a limited independent validation while still representing only one external dataset.",
        ]),
        ("III. FLIGHTREADY V2 METHODOLOGY", [
            "A. Context and evidence states. A project declares the task, target labels, acquisition assumptions, and applicable metrics. Each applicable metric is measured, unavailable, or not applicable. Unavailable evidence remains in the applicability denominator; it is never converted into a passing value. Each measured metric produces a normalized score q_i in [0,100], a positive weight w_i, provenance, and threshold evidence.",
            "B. Weighted geometric aggregation. Let A be the applicable metrics and M be the measured subset. The applicable and measured weight totals are W_A and W_M. Evidence coverage is the measured-weight fraction. For positive measured scores, the measured-quality summary is defined by (1). The denominator is the sum of measured weights, not a count of metrics. If any measured score with positive weight is zero, the implementation returns Q_M=0 directly because the logarithm is undefined and the geometric product contains a zero factor.",
            "EQ1",
            "The primary quality score equals Q_M only when coverage is exactly one; otherwise it is withheld. The measured-quality summary remains available but is labeled as partial evidence. Lower and upper score bounds are computed over A by assigning every unmeasured applicable metric a score of zero or 100, respectively. Consequently, any unmeasured positive-weight metric makes the strict lower geometric bound zero.",
            "C. Independent readiness gates. The fixed decision order is assessment error, measured blocking failure, unknown blocking evidence or coverage below 0.8, measured warning, and ready. A numeric summary therefore cannot override a blocking gate. The primary score requires complete applicable evidence even though the categorical decision uses a minimum coverage threshold. Algorithm 1 states the implementable procedure.",
            "D. Method freeze. Version flightready-v2.0 was frozen before independent result inspection. The integrity audit verified three recorded file hashes directly. The fourth manifest entry is a malformed 63-character string; inserting its omitted hexadecimal character reproduces the exact CRLF file hash, and LF normalization is byte-identical to the research-release Git blob. The original manifest remains unchanged. This is a documented manifest transcription defect, not a source-code or methodological change.",
            "E. Auditability and interpretation. Each exported scorecard retains the task context, methodology version, measured and unavailable metric states, individual weights and scores, coverage, measured-quality summary, primary-score availability, lower and upper bounds, warnings, blocking reasons, and provenance. This record is necessary because identical numeric summaries can have different meanings when their evidence coverage differs. In particular, Q_M summarizes only M; it is not an estimator of the unmeasured metrics, a calibrated probability of readiness, or a prediction of model accuracy. The categorical decision is likewise an engineering judgment under declared gates rather than a safety certification.",
            "ALGORITHM",
        ]),
        ("IV. EXPERIMENTAL METHODOLOGY", [
            "A. Historical studies and v2 reassessment. The tabular study trained 108 models across 18 controlled conditions, two regressors, and three seeds. Combined degradation reduced mean R-squared from 0.1635 to 0.0573 for histogram gradient boosting and from 0.1567 to 0.0720 for random forest. The VisDrone study used seven YOLO11n conditions and an untouched test-dev split. Clean mAP@0.5:0.95 was 0.03328; wrong-class, incomplete, and combined conditions reached 0.02928, 0.02808, and 0.02820. V2 reassessment reused these preserved outcomes and is method-development evidence rather than independent validation. Table I separates those stages.",
            "B. Independent UAVVaste dataset. UAVVaste contains 772 images and 3,718 audited COCO annotations for one rubbish category, with official 540/116/116 train/validation/test splits [7], [8]. The downloaded 3.01-GB archive matched the recorded MD5. All generated variants retained byte-identical official test files, and the audit found zero test-byte issues. Exact filename overlap was zero, although inferred source groups crossed all split pairs, which is a leakage threat.",
            "C. Controlled variants. Deterministic training-only variants used seed 20261112: Gaussian noise at 10%, 25%, and 40% prevalence with RGB sigma 25; horizontal 9-pixel motion blur at 25%; removal of half the boxes in affected images at 10%, 25%, and 40%; normalized center displacement of +0.08 at 25%; removal of 40% of training images; a combined 40% condition; and a repaired control byte/label equivalent to clean. Wrong-class corruption was inapplicable because the dataset has one valid class.",
            "D. Model protocol and run accounting. YOLO11n and YOLOv8n used pretrained Ultralytics weights [9], 832-pixel input, batch 32, deterministic execution, patience 10, two workers, and 40 fixed epochs; optimizer=auto selected AdamW. Validation selected the best checkpoint, and the official test set was used only for final evaluation. A 67-epoch coverage control approximately matched optimizer updates. The amended plan contained 19 runs: three clean and three combined seeds, single-seed primary defects, severity checks, one fixed-update control, and one YOLOv8n clean/combined pair. All 19 are complete. Two Gaussian-25 runs at seeds 42 and 93 satisfy the final protocol but predate the amended plan and are reported separately as supplemental observations.",
            "E. Evidence modes, baselines, and statistics. Reference-free assessment used only dataset-internal observations. The limited mode used one deterministic 10% trusted annotation audit. Full-oracle mode compared all training content with the clean reference. Baselines included schema and label validity, file completeness, unreadable images, exact duplicates, an absolute clarity proxy, annotation/image counts, and center-distribution shifts. A defect was detected when its score decreased by more than 0.1 point or its warning set changed relative to clean. Pearson and Spearman tests use eight related condition means. Two-sided p-values are descriptive; seed t intervals characterize training stochasticity rather than population or dataset-level generalization.",
            "F. Independence and run controls. The v2 definitions, normalizations, weights, thresholds, coverage rule, aggregation, and gate ordering were locked before the UAVVaste results were interpreted. Variant generation changed training content only. The official validation and test images and annotations were retained, and a byte audit reported no test-set changes. Run identifiers encode model, condition, seed, schedule, batch, and worker count; saved JSON records retain detector metrics, elapsed time, checkpoint identity, and protocol metadata. The two supplemental Gaussian-25 runs were not silently absorbed into plan completion: the amended-plan count remains 19, while their valid measurements are used only where the manuscript explicitly labels n=3 as one planned plus two supplemental observations.",
        ]),
        ("V. RESULTS", [
            "A. Readiness evidence. Table II reports the representative scorecards. Reference-free primary scores were withheld for every UAVVaste condition because evidence coverage was 55.6%. Their measured summaries remained between 99.48 and 99.85, demonstrating why a partial summary must not be presented as complete readiness. The full-oracle score fell from 99.83 for clean to 86.83 for combined degradation; the deterministic 10% audit produced 86.43. Figure 5 shows how trusted evidence changes what can be assessed.",
            "B. Defect sensitivity and baselines. Under the prespecified rule, reference-free assessment detected 1/10 controlled defects, while the deterministic 10% audit and full oracle each detected 10/10. Conventional baselines detected 5/10, mainly annotation-count and file-coverage changes. Figure 2 compares the methods. These sensitivity values measure assessor response to injected defects; they are not true-positive rates for naturally occurring field errors.",
            "C. Detector performance. Table III reports the primary YOLO11n conditions. The three-seed clean mean was 0.4790 mAP@0.5:0.95 (SD 0.0059; 95% t interval [0.4643, 0.4937]). Combined-40 averaged 0.4452 (SD 0.0044; interval [0.4341, 0.4562]), a matched-seed mean decrease of 7.05%. Gaussian-25 averaged 0.4747 (SD 0.0024; interval [0.4688, 0.4806]); only seed 17 belongs to the amended 19-run plan, while seeds 42 and 93 are supplemental. The paired Gaussian-25 difference interval [-0.0229, 0.0144] includes zero. Repair reproduced the matching clean seed at 0.4724. Figure 4 displays repeated-seed variability.",
            "D. Severity, update control, and architecture transfer. Gaussian results were 0.4682, 0.4747, and 0.4621 at 10%, 25%, and 40% prevalence; because the 25% mean includes two supplemental observations, this is not a monotonic replicated dose-response test. Missing-annotation results declined from 0.4687 at 10% to 0.4467 at 40%. The 40% coverage condition reached 0.4459 at fixed epochs and 0.4472 with approximately matched updates, so additional updates did not remove the observed deficit. YOLOv8n decreased from 0.4861 clean to 0.4454 combined (-8.36%). Table IV and Fig. 3 summarize these descriptive controls.",
            f"E. Statistical interpretation. Figure 1 relates full-oracle scores to the eight prespecified YOLO11n condition means. Pearson r={pearson['coefficient']:.3f} (p={pearson['p_value']:.4f}) and Spearman rho={spearman['coefficient']:.3f} (p={spearman['p_value']:.4f}) are nominally significant at alpha=.05. The tests are not independent predictive validation: conditions are related interventions on one source dataset, n=8 is small, and repaired duplicates clean bytes and labels. Excluding repaired changes Pearson to r={p7['coefficient']:.3f} (p={p7['p_value']:.4f}), which does not cross .05, while Spearman remains rho={s7['coefficient']:.3f} (p={s7['p_value']:.4f}). I therefore treat the correlations as descriptive condition-level association rather than calibration, causal evidence, or proof of dataset-to-dataset generalization.",
            "F. Score behavior by evidence mode. The clean reference-free summary was 99.70, but its primary score was unavailable; Gaussian-25 produced an even higher partial summary of 99.85 although the full-oracle and limited-audit scores were 96.77 and 95.81. Box displacement produced full-oracle and audit scores of 93.68 and 97.85, while coverage loss produced 94.32 and 93.52. These differences are expected because a deterministic subset and a complete reference expose different evidence. The combined condition yielded the lowest scores in both trusted-evidence modes (86.83 oracle and 86.43 audit), whereas repaired returned to the clean value of 99.83. The results demonstrate responsiveness under controlled access to trusted evidence; they do not show that one audit sample is an unbiased estimator of the full oracle.",
        ]),
        ("VI. DISCUSSION", [
            "The strongest supported contribution is evidence accounting rather than accuracy forecasting. FlightReady v2 prevents a near-perfect partial-evidence summary from masquerading as a complete readiness result. Trusted references or audits recover sensitivity to image corruption, label omission, box displacement, coverage loss, and combined corruption, but this sensitivity depends on evidence that practitioners may not possess.",
            "The limited audit detected all injected variants under the prespecified rule, yet it was one deterministic sample and should not be generalized to arbitrary audit fractions or sampling schemes. Likewise, conventional checks provide useful low-cost signals but missed Gaussian noise, motion blur, and displaced boxes in this experiment. These comparisons clarify capability boundaries rather than establish universal superiority.",
            "The historical reassessments are method-development evidence. UAVVaste adds an independently sourced domain, but the score-performance analysis still uses controlled conditions from one dataset. The positive association supports a hypothesis that lower readiness accompanies lower detector performance under this protocol. It does not establish a portable readiness-to-mAP function.",
            "The performance controls also separate several alternative explanations without eliminating them. Matching optimizer updates for 40% coverage improved mAP@0.5:0.95 only from 0.4459 to 0.4472, leaving a 5.32% deficit relative to the matching clean seed. The YOLOv8n combined result decreased by 8.36%, which is directionally consistent with YOLO11n, but each YOLOv8n condition has one seed. Repair exactly reproduced clean data and the matching 0.4724 detector outcome, supporting pipeline determinism for that control rather than proving determinism for every training run.",
            "Practically, FlightReady should be read as a structured argument about evidence sufficiency. A practitioner can inspect which criteria were applicable, why a measurement was unavailable, whether a gate failed, and how much weight was actually measured. When trusted references are absent, the correct output may be insufficient evidence even when every available technical check passes. That behavior is a substantive result of the project, not a missing implementation feature.",
        ]),
        ("VII. LIMITATIONS AND THREATS TO VALIDITY", [
            "External validity is limited to one independently sourced, single-class UAV dataset and two related YOLO detector families. Independent wrong-class validation remains unavailable. The official image-level split contains inferred filename source groups across split pairs, which may include adjacent frames and inflate detector performance. I retained the official split because stronger grouping metadata were unavailable.",
            "Construct validity is limited by the evidence modes. Full-oracle comparison is experimentally informative but operationally unrealistic. The deterministic 10% audit was not replicated. Gaussian noise can increase a Laplacian sharpness proxy, showing that absolute proxies are content-sensitive. Conventional baselines and FlightReady do not have identical evidence access, so sensitivity comparisons describe practical detection capability rather than a controlled algorithm contest.",
            "Statistical validity is limited by single-seed defect conditions, two supplemental Gaussian observations outside the amended plan, related condition means, and small n. Confidence intervals with three seeds describe training variability only. Metric weights and thresholds are engineering judgments rather than learned utilities. No result implies causal prediction, dataset-level generalization, aviation safety, or certified operational readiness.",
            "Reproducibility is also bounded by the retained software and hardware environment. Pretrained weights and saved checkpoints are identified in the experiment records, but future library, driver, or accelerator changes can affect exact training trajectories. The frozen source identity is recoverable despite the documented 63-character manifest transcription defect, yet the defect itself shows why manifests should be machine-validated at freeze time. The read-only verification script checks the saved evidence; it does not recreate raw training computation from first principles.",
        ]),
        ("VIII. CONCLUSION", [
            "I developed and evaluated FlightReady v2 as a context-aware, evidence-aware readiness pipeline. The 19 completed amended-plan runs, two separately labeled supplemental observations, historical studies, and v2 reassessments support transparent score withholding and reference-assisted controlled-defect sensitivity. They do not establish universal prediction of downstream performance. I conclude that FlightReady v2 is most defensible as an auditable decision-support framework that exposes what is measured, what is missing, and which gates prevent a readiness claim.",
        ]),
        ("IX. REPRODUCIBILITY", [
            "The artifact package contains the frozen-method manifest, experiment configuration, dataset audit, protocol amendments, per-run JSON records, model checkpoint hashes, scorecards, analysis outputs, and figure-generation inputs. An independent read-only verification recalculated run accounting, group means, paired changes, t intervals, correlations, evidence coverage, geometric scores, categorical decisions, detection sensitivity, and baseline detection from the saved artifacts. The original manifest is preserved with its documented 63-character transcription defect.",
            "The verification succeeded for all 19 planned tags and identified exactly the two declared supplemental tags. Every saved condition-summary value, three-seed interval, paired change, correlation coefficient and p-value, representative scorecard, detection count, and conventional-baseline result matched the independent recalculation. These checks establish internal consistency between the retained artifacts and this manuscript; they do not replace replication on another independently sourced dataset.",
        ]),
    ]


REFERENCES = [
    "[1] S. Afzal, R. C., M. Kesarwani, S. Mehta, and H. Patel, \"Data Readiness Report,\" arXiv:2010.07213, 2020.",
    "[2] M. Priestley, F. O'Donnell, and E. Simperl, \"A Survey of Data Quality Requirements That Matter in ML Development Pipelines,\" J. Data Inf. Quality, vol. 15, no. 2, pp. 1-39, 2023, doi: 10.1145/3592616.",
    "[3] Y. Gong, G. Liu, Y. Xue, R. Li, and L. Meng, \"A survey on dataset quality in machine learning,\" Inf. Softw. Technol., vol. 162, Art. no. 107268, 2023, doi: 10.1016/j.infsof.2023.107268.",
    "[4] N. Sambasivan et al., \"Everyone wants to do the model work, not the data work: Data cascades in high-stakes AI,\" in Proc. CHI, 2021, doi: 10.1145/3411764.3445518.",
    "[5] T. Gebru et al., \"Datasheets for datasets,\" Commun. ACM, vol. 64, no. 12, pp. 86-92, 2021, doi: 10.1145/3458723.",
    "[6] P. Zhu et al., \"VisDrone-DET2018: The Vision Meets Drone Object Detection in Image Challenge Results,\" in Proc. ECCV Workshops, 2018.",
    "[7] M. Kraft et al., \"Autonomous, Onboard Vision-Based Trash and Litter Detection in Low Altitude Aerial Images Collected by an Unmanned Aerial Vehicle,\" Remote Sens., vol. 13, no. 5, Art. no. 965, 2021, doi: 10.3390/rs13050965.",
    "[8] M. Kraft et al., \"UAVVaste,\" Zenodo, 2023, doi: 10.5281/zenodo.8214061.",
    "[9] Ultralytics, \"Ultralytics YOLO Documentation.\" [Online]. Available: https://docs.ultralytics.com/. Accessed: Nov. 13, 2026.",
]


def docx_columns(section, count, space_twips=260):
    cols = section._sectPr.first_child_found_in("w:cols")
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(space_twips))


def style_docx(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_after = Pt(2)


def docx_heading(doc, text, centered=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if centered else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)


def docx_para(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.14)
    p.paragraph_format.space_after = Pt(2)
    p.add_run(text)


def docx_table(doc, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(rows[0]))
    table.style = "Table Grid"
    for j, value in enumerate(rows[0]):
        cell = table.rows[0].cells[j]
        cell.text = str(value)
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "D9E2F3")
        cell._tc.get_or_add_tcPr().append(shade)
    for row in rows[1:]:
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cells[j].text = str(value)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(6.4)
                    if row is table.rows[0]:
                        run.bold = True
    return table


def build_docx(ev, equation_figure):
    path = OUT / f"{STEM}.docx"
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.55); sec.bottom_margin = Inches(0.55)
    sec.left_margin = Inches(0.58); sec.right_margin = Inches(0.58)
    docx_columns(sec, 1)
    style_docx(doc)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE); r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(17)
    for line in (AUTHOR, ROLE):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
        rr = p.add_run(line); rr.font.name = "Times New Roman"; rr.font.size = Pt(10)
    abstract = (
        "I present FlightReady v2, an evidence-aware data-readiness pipeline for intelligent UAV systems. "
        "The frozen method retains unavailable evidence in coverage, separates weighted geometric aggregation from readiness gates, and withholds the primary score when applicable evidence is incomplete. "
        "I report historical tabular and VisDrone studies, their v2 reassessment, and independent UAVVaste validation. All 19 amended-plan runs are complete; two additional valid Gaussian-25 observations are labeled supplemental. "
        "Reference-free scores were withheld at 55.6% coverage. Under a prespecified detection rule, sensitivity was 10% reference-free and 100% with either a deterministic 10% audit or full oracle. "
        "Across eight related condition means, Pearson r=.767 (p=.0263) and Spearman rho=.838 (p=.0093), but these descriptive associations do not independently validate prediction. The evidence supports transparent evidence accounting and reference-assisted defect sensitivity, not a universal readiness-to-accuracy predictor."
    )
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("Abstract-").bold = True; p.add_run(abstract)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("Index Terms-").bold = True; p.add_run("data readiness, data quality, evidence coverage, UAV, object detection, reproducibility")

    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    sec2.top_margin = Inches(0.55); sec2.bottom_margin = Inches(0.55)
    sec2.left_margin = Inches(0.58); sec2.right_margin = Inches(0.58)
    docx_columns(sec2, 2)

    table_map = {
        "IV. EXPERIMENTAL METHODOLOGY": ("TABLE I", "Development and validation evidence", ev["history_rows"]),
        "V. RESULTS": ("TABLE II", "FlightReady evidence results", ev["quality_rows"]),
    }
    for heading, paragraphs in sections(ev):
        docx_heading(doc, heading)
        for text in paragraphs:
            if text == "EQ1":
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(equation_figure), width=Inches(3.15))
            elif text == "ALGORITHM":
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run("Algorithm 1: FlightReady v2 decision procedure").bold = True
                docx_table(doc, [
                    ["Step", "Operation"],
                    ["1", "Declare task context and applicable metric set A."],
                    ["2", "Measure available evidence M; retain unavailable evidence."],
                    ["3", "Compute c, Q_M, bounds, warnings, and blocking states."],
                    ["4", "Withhold Q unless M=A; apply fixed gate order."],
                    ["5", "Export scorecard, provenance, metric evidence, and hashes."],
                ])
            else:
                docx_para(doc, text)
        if heading in table_map:
            number, caption, rows = table_map[heading]
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = p.add_run(f"{number}\n{caption}"); rr.bold = True; rr.font.size = Pt(7)
            docx_table(doc, rows)
        if heading == "V. RESULTS":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = p.add_run("TABLE III\nPrimary YOLO11n results"); rr.bold = True; rr.font.size = Pt(7)
            docx_table(doc, ev["primary_rows"])
            p = doc.add_paragraph("*Gaussian-25 n=3 includes one planned run and two supplemental observations.")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs: run.font.size = Pt(6.5)
            for img_name, cap in [
                ("01_readiness_vs_performance.png", "Fig. 1. Full-oracle score versus condition-mean performance; the association is descriptive."),
                ("07_method_detection_comparison.png", "Fig. 2. Controlled-defect detection by assessment method and conventional baselines."),
                ("02_performance_vs_severity.png", "Fig. 3. Single-seed severity observations for Gaussian noise and missing annotations."),
                ("08_seed_variability.png", "Fig. 4. Repeated-seed variability; Gaussian-25 includes two supplemental observations."),
                ("04_reference_evidence_detection.png", "Fig. 5. Defect detection changes with trusted reference evidence."),
            ]:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(FIG / img_name), width=Inches(3.15))
                cp = doc.add_paragraph(cap); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in cp.runs: run.font.size = Pt(7)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = p.add_run("TABLE IV\nControls and architecture transfer"); rr.bold = True; rr.font.size = Pt(7)
            docx_table(doc, ev["extended_rows"])

    docx_heading(doc, "REFERENCES")
    for ref in REFERENCES:
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.first_line_indent = Inches(-0.15); p.paragraph_format.space_after = Pt(1)
        r = p.add_run(ref); r.font.name = "Times New Roman"; r.font.size = Pt(7)
    doc.core_properties.author = ""
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "Anonymous IEEE IV 2027 initial-submission draft"
    doc.save(path)
    return path


def pdf_styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="TNR", fontSize=9.5,
                          leading=10.9, alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=3.5)
    heading = ParagraphStyle("heading", parent=body, fontName="TNR-Bold", fontSize=9.2,
                             leading=10.7, alignment=TA_CENTER, firstLineIndent=0, spaceBefore=5.5, spaceAfter=3.2)
    tablecap = ParagraphStyle("tablecap", parent=body, fontName="TNR-Bold", fontSize=6.7,
                              leading=7.4, alignment=TA_CENTER, firstLineIndent=0, spaceBefore=3, spaceAfter=2)
    small = ParagraphStyle("small", parent=body, fontSize=6.45, leading=7.3, firstLineIndent=0, spaceAfter=1.5)
    title = ParagraphStyle("title", fontName="TNR-Bold", fontSize=17, leading=19, alignment=TA_CENTER, spaceAfter=5)
    author = ParagraphStyle("author", fontName="TNR", fontSize=9.5, leading=10.5, alignment=TA_CENTER, spaceAfter=0)
    abstract = ParagraphStyle("abstract", parent=body, fontSize=9.0, leading=10.1, firstLineIndent=0, spaceAfter=3)
    equation = ParagraphStyle("equation", parent=body, fontName="TNR-Italic", fontSize=8.3,
                              leading=10, alignment=TA_CENTER, firstLineIndent=0, spaceAfter=4)
    return body, heading, tablecap, small, title, author, abstract, equation


def pdf_table(rows, col_widths, font_size=5.8):
    data = [[Paragraph(str(v), ParagraphStyle("tc", fontName="TNR", fontSize=font_size,
                                               leading=font_size + 0.8, alignment=TA_LEFT)) for v in row] for row in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
        ("FONTNAME", (0, 0), (-1, -1), "TNR"),
        ("FONTNAME", (0, 0), (-1, 0), "TNR-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#777777")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    return t


def build_pdf(ev, equation_figure):
    path = OUT / f"{STEM}.pdf"
    body, heading, tablecap, small, title, author, abstract, equation = pdf_styles()
    page_w, page_h = letter
    margin = 0.625 * inch; gap = 0.25 * inch
    col_w = (page_w - 2 * margin - gap) / 2
    top_h = 2.72 * inch; body_top = page_h - margin - top_h - 0.12 * inch
    first_frames = [
        Frame(margin, body_top + 0.12 * inch, page_w - 2 * margin, top_h, id="title", showBoundary=0),
        Frame(margin, margin, col_w, body_top - margin, id="first_left", showBoundary=0),
        Frame(margin + col_w + gap, margin, col_w, body_top - margin, id="first_right", showBoundary=0),
    ]
    later_frames = [
        Frame(margin, margin, col_w, page_h - 2 * margin, id="left", showBoundary=0),
        Frame(margin + col_w + gap, margin, col_w, page_h - 2 * margin, id="right", showBoundary=0),
    ]
    def footer(canvas, doc):
        canvas.setAuthor(""); canvas.setTitle(TITLE)
        canvas.setSubject("Anonymous IEEE IV 2027 initial-submission draft")
        canvas.setKeywords("data readiness, data quality, UAV, object detection")
        canvas.saveState(); canvas.setFont("TNR", 7)
        canvas.drawCentredString(page_w / 2, 0.30 * inch, str(doc.page)); canvas.restoreState()
    doc = BaseDocTemplate(str(path), pagesize=letter, leftMargin=margin, rightMargin=margin,
                          topMargin=margin, bottomMargin=margin)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=first_frames, onPage=footer),
        PageTemplate(id="later", frames=later_frames, onPage=footer),
    ])
    abstract_text = (
        "I present FlightReady v2, an evidence-aware data-readiness pipeline for intelligent UAV systems. "
        "The frozen method retains unavailable evidence in coverage, separates weighted geometric aggregation from readiness gates, and withholds the primary score when applicable evidence is incomplete. "
        "I report historical tabular and VisDrone studies, their v2 reassessment, and independent UAVVaste validation. All 19 amended-plan runs are complete; two additional valid Gaussian-25 observations are labeled supplemental. "
        "Reference-free scores were withheld at 55.6% coverage. Under a prespecified detection rule, sensitivity was 10% reference-free and 100% with either a deterministic 10% audit or full oracle. "
        "Across eight related condition means, Pearson r=.767 (p=.0263) and Spearman rho=.838 (p=.0093), but these descriptive associations do not independently validate prediction. The evidence supports transparent evidence accounting and reference-assisted defect sensitivity, not a universal readiness-to-accuracy predictor."
    )
    flow = [
        NextPageTemplate("later"), Paragraph(TITLE, title), Paragraph(AUTHOR, author), Paragraph(ROLE, author),
        Spacer(1, 5), Paragraph("<b>Abstract-</b> " + abstract_text, abstract),
        Paragraph("<b>Index Terms-</b> data readiness, data quality, evidence coverage, UAV, object detection, reproducibility", abstract),
        FrameBreak(),
    ]
    table_map = {
        "IV. EXPERIMENTAL METHODOLOGY": ("TABLE I", "DEVELOPMENT AND VALIDATION EVIDENCE", ev["history_rows"], [0.62*inch, 0.64*inch, 0.91*inch, 0.88*inch]),
        "V. RESULTS": ("TABLE II", "FLIGHTREADY EVIDENCE RESULTS", ev["quality_rows"], [0.72*inch, 0.43*inch, 0.43*inch, 0.43*inch, 0.43*inch, 0.61*inch]),
    }
    for sec_heading, paragraphs in sections(ev):
        flow.append(Paragraph(sec_heading, heading))
        for text in paragraphs:
            if text == "EQ1":
                flow.append(Image(str(equation_figure), width=col_w*0.97, height=0.42*inch))
            elif text == "ALGORITHM":
                algo = [
                    ["Step", "Operation"],
                    ["1", "Declare task context and applicable metric set A."],
                    ["2", "Measure M; retain every unavailable applicable item."],
                    ["3", "Compute coverage, Q_M, bounds, warnings, and blocks."],
                    ["4", "Withhold Q unless M=A; apply fixed gate order."],
                    ["5", "Export evidence, provenance, and frozen hashes."],
                ]
                flow.append(Paragraph("ALGORITHM 1. FLIGHTREADY V2 DECISION PROCEDURE", tablecap))
                flow.append(pdf_table(algo, [0.3*inch, 2.72*inch], 5.9))
            else:
                flow.append(Paragraph(text, body))
        if sec_heading in table_map:
            number, caption, rows, widths = table_map[sec_heading]
            flow.extend([Paragraph(f"{number}. {caption}", tablecap), pdf_table(rows, widths, 5.45)])
        if sec_heading == "V. RESULTS":
            flow.extend([
                Paragraph("TABLE III. PRIMARY YOLO11N RESULTS", tablecap),
                pdf_table(ev["primary_rows"], [0.86*inch, 0.24*inch, 0.63*inch, 0.45*inch, 0.85*inch], 5.55),
                Paragraph("*Gaussian-25 n=3 contains one amended-plan run and two supplemental observations.", small),
                KeepTogether([Image(str(FIG / "01_readiness_vs_performance.png"), width=col_w, height=col_w*0.60),
                              Paragraph("Fig. 1. Full-oracle score versus condition-mean performance. This is descriptive, not independently validated prediction.", small)]),
                KeepTogether([Image(str(FIG / "07_method_detection_comparison.png"), width=col_w, height=col_w*0.60),
                              Paragraph("Fig. 2. Controlled-defect detection by evidence mode and conventional baselines.", small)]),
                KeepTogether([Image(str(FIG / "02_performance_vs_severity.png"), width=col_w, height=col_w*0.60),
                              Paragraph("Fig. 3. Single-seed severity observations for Gaussian and missing-annotation variants.", small)]),
                Paragraph("TABLE IV. CONTROLS AND ARCHITECTURE TRANSFER", tablecap),
                pdf_table(ev["extended_rows"], [1.12*inch, 0.54*inch, 0.58*inch, 0.79*inch], 5.55),
                KeepTogether([Image(str(FIG / "08_seed_variability.png"), width=col_w, height=col_w*0.58),
                              Paragraph("Fig. 4. Seed variability for conditions with repeated observations; Gaussian-25 includes two supplemental seeds.", small)]),
                KeepTogether([Image(str(FIG / "04_reference_evidence_detection.png"), width=col_w, height=col_w*0.58),
                              Paragraph("Fig. 5. Controlled-defect detection changes with trusted reference evidence.", small)]),
            ])
    flow.append(Paragraph("REFERENCES", heading))
    for ref in REFERENCES:
        flow.append(Paragraph(ref, small))
    def embedded_canvas(*args, **kwargs):
        kwargs.update(initialFontName="TNR", pdfVersion=(1, 4), pageCompression=1)
        return pdfcanvas.Canvas(*args, **kwargs)
    doc.build(flow, canvasmaker=embedded_canvas)
    return path


def tex_escape(text):
    return (text.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("{", "\\{").replace("}", "\\}"))


def tex_with_citations(text):
    escaped = tex_escape(text)
    replacements = {
        "[1]-[3]": r"\cite{r1,r2,r3}",
        "[4], [5]": r"\cite{r4,r5}",
        "[7], [8]": r"\cite{r7,r8}",
        "[6]": r"\cite{r6}",
        "[9]": r"\cite{r9}",
    }
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def build_tex(ev):
    path = OUT / f"{STEM}.tex"
    sec_parts = []
    for heading, paragraphs in sections(ev):
        if heading.startswith("ACKNOWLEDGMENT"):
            sec_parts.append("\\section*{Acknowledgment and AI-Assistance Disclosure}")
        else:
            title = heading.split(". ", 1)[-1].title()
            sec_parts.append(f"\\section{{{tex_escape(title)}}}")
        for text in paragraphs:
            if text == "EQ1":
                sec_parts.append(r"\begin{align}W_A&=\sum_{i\in A}w_i, & W_M&=\sum_{i\in M}w_i, & c&=\frac{W_M}{W_A},\\ Q_M&=100\exp\!\left(\frac{1}{W_M}\sum_{i\in M}w_i\ln\frac{q_i}{100}\right).\end{align}")
            elif text == "ALGORITHM":
                sec_parts.append(r"\begin{figure}[t]\small\textbf{Algorithm 1: FlightReady v2 decision procedure}\par Declare context and $A$; measure $M$ while retaining unavailable evidence; compute $c$, $Q_M$, bounds and gates; withhold $Q$ unless $M=A$; export provenance and hashes.\end{figure}")
            else:
                sec_parts.append(tex_with_citations(text))
    refs = "\n".join(
        f"\\bibitem{{r{i}}} {tex_escape(ref.split('] ', 1)[1])}" for i, ref in enumerate(REFERENCES, 1)
    )
    def rows_for(rows):
        return "\n".join(" & ".join(tex_escape(str(v)) for v in row) + r" \\" for row in rows[1:])
    history_rows = rows_for(ev["history_rows"])
    quality_rows = rows_for(ev["quality_rows"])
    primary_rows = rows_for(ev["primary_rows"])
    extended_rows = rows_for(ev["extended_rows"])
    abstract_text = (
        "I present FlightReady v2, an evidence-aware data-readiness pipeline for intelligent UAV systems. "
        "The frozen method retains unavailable evidence in coverage, separates weighted geometric aggregation from readiness gates, and withholds the primary score when applicable evidence is incomplete. "
        "All 19 amended-plan runs are complete; two additional valid Gaussian-25 observations are labeled supplemental. "
        "Reference-free scores were withheld at 55.6% coverage. Sensitivity was 10% reference-free and 100% with either a deterministic 10% audit or full oracle. "
        "Across eight related condition means, Pearson r=.767 (p=.0263) and Spearman rho=.838 (p=.0093), but these descriptive associations do not independently validate prediction."
    )
    tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\title{{{tex_escape(TITLE)}}}
\author{{\IEEEauthorblockN{{{AUTHOR}}}\IEEEauthorblockA{{{ROLE}}}}}
% Camera-ready identity (activate only after acceptance and author confirmation):
% Aditya Pratap Singh, Undergraduate Researcher, The Pennsylvania State University.
% Department, email, ORCID, and any coauthors remain intentionally unspecified.
% Camera-ready AI-assistance disclosure, subject to IEEE IV 2027 and institutional review:
% OpenAI Codex provided substantive assistance with research software development,
% automated experiment orchestration, analysis scripting, and manuscript preparation.
% The researcher directed the study, verified the retained evidence, and accepts
% responsibility for the manuscript. This disclosure does not resolve the IV 2027
% prohibition on LLM-generated manuscripts; a human-authored rewrite may be required.
\begin{{document}}
\maketitle
\begin{{abstract}}{tex_escape(abstract_text)}\end{{abstract}}
\begin{{IEEEkeywords}}data readiness, data quality, evidence coverage, UAV, object detection, reproducibility\end{{IEEEkeywords}}
{chr(10).join(sec_parts)}
\begin{{table*}}[t]
\caption{{Development and validation evidence.}}
\centering\scriptsize
\begin{{tabular}}{{p{{0.13\textwidth}}p{{0.19\textwidth}}p{{0.28\textwidth}}p{{0.27\textwidth}}}}
\toprule Study & Scope & Readiness-performance result & Interpretation \\\midrule
{history_rows}
\bottomrule\end{{tabular}}
\end{{table*}}
\begin{{table}}[t]
\caption{{FlightReady evidence results.}}
\centering\scriptsize
\begin{{tabular}}{{lccccc}}\toprule Condition & RF & Coverage & Oracle & Audit & Decision \\\midrule
{quality_rows}
\bottomrule\end{{tabular}}
\end{{table}}
\begin{{table}}[t]
\caption{{Primary YOLO11n results.}}
\centering\scriptsize
\begin{{tabular}}{{lcccc}}\toprule Condition & $n$ & mAP50-95 & SD & Relative change \\\midrule
{primary_rows}
\bottomrule\end{{tabular}}
\par\vspace{{1mm}}\raggedright\scriptsize *Gaussian-25 $n=3$ contains one amended-plan run and two supplemental observations.
\end{{table}}
\begin{{figure}}[t]\centering\includegraphics[width=\columnwidth]{{figures/01_readiness_vs_performance.png}}\caption{{Full-oracle score versus condition-mean performance. The relationship is descriptive, not independently validated prediction.}}\end{{figure}}
\begin{{figure}}[t]\centering\includegraphics[width=\columnwidth]{{figures/07_method_detection_comparison.png}}\caption{{Controlled-defect detection by evidence mode and conventional baselines.}}\end{{figure}}
\begin{{figure}}[t]\centering\includegraphics[width=\columnwidth]{{figures/02_performance_vs_severity.png}}\caption{{Single-seed severity observations for Gaussian and missing-annotation variants.}}\end{{figure}}
\begin{{table}}[t]
\caption{{Controls and architecture transfer.}}
\centering\scriptsize
\begin{{tabular}}{{lccc}}\toprule Comparison & Reference & Result & Change/interpretation \\\midrule
{extended_rows}
\bottomrule\end{{tabular}}
\end{{table}}
\begin{{figure}}[t]\centering\includegraphics[width=\columnwidth]{{figures/08_seed_variability.png}}\caption{{Seed variability for repeated observations; Gaussian-25 includes two supplemental seeds.}}\end{{figure}}
\begin{{figure}}[t]\centering\includegraphics[width=\columnwidth]{{figures/04_reference_evidence_detection.png}}\caption{{Controlled-defect detection changes with trusted reference evidence.}}\end{{figure}}
\begin{{thebibliography}}{{10}}
{refs}
\end{{thebibliography}}
\end{{document}}
"""
    path.write_text(tex, encoding="utf-8")
    return path


def build_changelog():
    path = OUT / f"{STEM}_Change_Log.md"
    text = """# FlightReady IEEE IV 2027 final-draft change log

## Corrections completed

- Corrected the weighted geometric mean to normalize by the total measured weight: `Q_M = 100 exp[(1/W_M) sum w_i ln(q_i/100)]`.
- Defined applicable and measured weight totals, evidence coverage, score withholding, strict score bounds, and log-space metric influences.
- Resolved the significance contradiction. The eight-condition Pearson and Spearman results are nominally significant at alpha 0.05, but they are reported only as descriptive condition-level associations. Excluding the repaired duplicate control changes Pearson to `p=0.0573`; no predictive-validation claim is made.
- Made the initial-submission manuscript anonymous in accordance with the official IV 2027 double-blind instructions. The LaTeX source contains only commented camera-ready identity information: **Aditya Pratap Singh, Undergraduate Researcher, The Pennsylvania State University**. No department, email, ORCID, professor, or coauthor was invented.
- Distinguished the **19 completed amended-plan runs** from **two supplemental Gaussian-25 observations** in the abstract, methods, results, table note, and figure caption.
- Expanded related work, methodology, experimental protocol, baselines, results, discussion, limitations, conclusion, reproducibility record, tables, and figures.
- Replaced unverified or unsuitable citations with traceable scholarly sources and official dataset/software sources.
- Added a commented camera-ready AI-assistance disclosure to the editable source and documented the stricter IV 2027 LLM prohibition in the verification report. The anonymous initial-submission PDF contains no acknowledgment, as required by the double-blind instructions.
- Preserved the frozen FlightReady v2 methodology and verified results; no experiment was run and no method file was changed.

## Integrity-audit notice

Three of the four SHA-256 values in `frozen_methodology.json` match their current files exactly. The recorded `backend/readiness_engine.py` value has only 63 hexadecimal characters. Inserting the omitted `e` produces the exact current CRLF-file hash, and the LF-normalized current file is byte-identical to the research-release Git blob. The original manifest remains unchanged. The integrity audit therefore resolves this as a manifest transcription defect, not a source or methodology change.

## Format and track determination

The manuscript is formatted as a regular/contributed IEEE IV 2027 paper. The official author page states an expected length of six pages and permits up to eight pages with charges for the two extra pages. It also requires US Letter PDF, searchable text, embedded fonts, and anonymous initial submission.

## Unresolved conference-policy issue

The official IV 2027 author instructions state that manuscripts generated by LLMs are prohibited and allow only light LLM editing for grammar and spelling. This draft received substantive Codex assistance. A disclosure alone does not resolve the local prohibition. The researcher and professor must decide whether a fully human-authored rewrite and institutional review are required before submission. For this reason, the package is a professor-review draft and is **not represented as submission-ready**.

## Required pre-submission actions

1. Resolve the IV 2027 LLM-generation prohibition with the professor and applicable institutional guidance; perform a human-authored rewrite if required.
2. Confirm the final author list and add verified camera-ready affiliation/contact details only at the appropriate stage.
3. Compile the LaTeX source in the official IEEE environment and run the conference PDF checker.
4. Perform scientific and authorship review. This package has not been submitted, committed, or pushed.

## Official guidance checked

- IEEE conference templates: https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
- IEEE submission and AI-generated-content policy: https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/
- IV 2027 author instructions: https://ieee-iv.org/2027/contributions/information-for-authors/
- IV 2027 call for papers: https://ieee-iv.org/2027/contributions/call-for-papers/
- IV 2027 PaperCept portal: https://its.papercept.net/conferences/scripts/start.pl
"""
    path.write_text(text, encoding="utf-8")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, quality, frozen, summary = load_evidence()
    ev = evidence_model(stats, quality, frozen, summary)
    stage_latex_figures()
    equation_figure = make_equation_figure()
    outputs = [build_docx(ev, equation_figure), build_pdf(ev, equation_figure), build_tex(ev), build_changelog()]
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
