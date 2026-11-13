"""Apply frozen FlightReady v2 scorecards to every independent UAVVaste variant."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.readiness_v2 import build_scorecard, fraction_metric

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
PREPARED = EXP / "dataset" / "prepared"
CONFIG = json.loads((EXP / "experiment_config.json").read_text(encoding="utf-8"))
SPLITS = json.loads((EXP / "dataset" / "source" / "annotations" / "train_val_test_distribution_file.json").read_text(encoding="utf-8"))
OUT = EXP / "quality_results"
CLARITY_THRESHOLD = 100.0


FILE_DIGESTS = {}
IMAGE_ANALYSIS = {}


def file_identity(path: Path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def digest(path: Path) -> str:
    identity = file_identity(path)
    if identity in FILE_DIGESTS:
        return FILE_DIGESTS[identity]
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    FILE_DIGESTS[identity] = h.hexdigest()
    return FILE_DIGESTS[identity]


def image_analysis(path: Path):
    identity = file_identity(path)
    if identity in IMAGE_ANALYSIS:
        return IMAGE_ANALYSIS[identity]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        result = (False, False)
    else:
        scale = min(1.0, 640.0 / max(image.shape))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = (True, float(cv2.Laplacian(image, cv2.CV_64F).var()) < CLARITY_THRESHOLD)
    IMAGE_ANALYSIS[identity] = result
    return result


def labels(path: Path):
    rows = []
    malformed = False
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            values = [float(v) for v in line.split()]
            if len(values) != 5 or values[0] != int(values[0]) or not all(np.isfinite(values)) or any(v < 0 or v > 1 for v in values[1:]):
                malformed = True
            rows.append(values)
    except Exception:
        malformed = True
    return rows, malformed


def iou(a, b):
    def corners(row):
        _, x, y, w, h = row
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2
    ax1, ay1, ax2, ay2 = corners(a)
    bx1, by1, bx2, by2 = corners(b)
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def observations(variant: str):
    root = PREPARED / variant
    names = [p.name for p in (root / "images" / "train").iterdir() if p.is_file()]
    unreadable = malformed = missing_labels = 0
    low_clarity = 0
    hashes = []
    box_count = 0
    centers = []
    for name in names:
        image_path = root / "images" / "train" / name
        readable, low = image_analysis(image_path)
        if not readable:
            unreadable += 1
        elif low:
            low_clarity += 1
        hashes.append(digest(image_path))
        label_path = root / "labels" / "train" / f"{Path(name).stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        rows, bad = labels(label_path)
        malformed += int(bad)
        box_count += len(rows)
        centers.extend((r[1], r[2]) for r in rows if len(r) == 5)
    duplicate_count = len(hashes) - len(set(hashes))
    return {
        "variant": variant, "training_images": len(names), "unreadable_images": unreadable,
        "malformed_label_files": malformed, "missing_label_files": missing_labels,
        "low_clarity_images": low_clarity, "duplicate_images_after_first": duplicate_count,
        "box_count": box_count,
        "mean_center_x": float(np.mean([x for x, _ in centers])) if centers else None,
        "mean_center_y": float(np.mean([y for _, y in centers])) if centers else None,
    }


def reference_metrics(variant: str, audited_names: set[str] | None = None):
    clean = PREPARED / "clean"
    candidate = PREPARED / variant
    names = audited_names if audited_names is not None else set(SPLITS["train"])
    image_different = missing_images = clean_boxes = missing_boxes = compared_boxes = geometry_bad = 0
    for name in sorted(names):
        clean_image = clean / "images" / "train" / name
        candidate_image = candidate / "images" / "train" / name
        if not candidate_image.exists():
            missing_images += 1
            continue
        image_different += int(digest(candidate_image) != digest(clean_image))
        ref_rows, _ = labels(clean / "labels" / "train" / f"{Path(name).stem}.txt")
        cand_rows, _ = labels(candidate / "labels" / "train" / f"{Path(name).stem}.txt")
        clean_boxes += len(ref_rows)
        unmatched = set(range(len(ref_rows)))
        for row in cand_rows:
            matches = [(iou(row, ref_rows[j]), j) for j in unmatched if int(row[0]) == int(ref_rows[j][0])]
            best_iou, best = max(matches, default=(0.0, None))
            compared_boxes += 1
            if best_iou < 0.5:
                geometry_bad += 1
            elif best is not None:
                unmatched.remove(best)
        missing_boxes += len(unmatched)
    available = len(names) - missing_images
    return {
        "image_difference_fraction": image_different / available if available else 0.0,
        "training_image_missing_fraction": missing_images / len(names) if names else 0.0,
        "annotation_missing_fraction": missing_boxes / clean_boxes if clean_boxes else 0.0,
        "geometry_disagreement_fraction": geometry_bad / compared_boxes if compared_boxes else 0.0,
        "reference_image_count": len(names), "reference_box_count": clean_boxes,
    }


def scorecard(obs, ref=None, *, mode: str, sample_fraction: float | None = None):
    n = max(obs["training_images"], 1)
    limitation = f"Observed on supplied training data; reference mode={mode}."
    metrics = [
        fraction_metric("annotation_validity", "annotation_quality", "Malformed label files", obs["malformed_label_files"] / n, weight=1.0, limitation=limitation, gate="block", maximum=0.0),
        fraction_metric("label_availability", "annotation_quality", "Missing label files", obs["missing_label_files"] / n, weight=1.0, limitation=limitation, gate="block", maximum=0.0),
        fraction_metric("image_readability", "integrity", "Unreadable images", obs["unreadable_images"] / n, weight=1.0, limitation=limitation, gate="block", maximum=0.0),
        fraction_metric("exact_duplicates", "integrity", "Exact duplicate images", obs["duplicate_images_after_first"] / n, weight=1.0, limitation="Exact byte duplicates only.", gate="warning", maximum=0.02),
        fraction_metric("clarity_proxy", "signal_quality", "Images below frozen Laplacian threshold", obs["low_clarity_images"] / n, weight=1.0, limitation="Absolute Laplacian-variance proxy; noise can increase apparent sharpness.", gate="warning", maximum=0.30),
    ]
    ref_limitation = "Unavailable without a trusted clean reference."
    for metric_id, dimension, label, key in [
        ("reference_image_fidelity", "signal_quality", "Images differing from trusted reference", "image_difference_fraction"),
        ("reference_annotation_completeness", "annotation_quality", "Missing trusted-reference objects", "annotation_missing_fraction"),
        ("reference_box_geometry", "annotation_quality", "Boxes without IoU>=0.5 trusted match", "geometry_disagreement_fraction"),
        ("acquisition_coverage", "statistical_adequacy", "Expected training images absent", "training_image_missing_fraction"),
    ]:
        metrics.append(fraction_metric(metric_id, dimension, label, None if ref is None else ref[key], weight=1.0,
            evidence_type="trusted_reference" if ref is not None else "unavailable_reference",
            reference_id="UAVVaste-clean-training" if ref is not None else None,
            limitation=ref_limitation if ref is None else (f"Deterministic {sample_fraction:.0%} audit sample." if sample_fraction else "Full oracle-assisted training comparison."),
            gate="warning", maximum=0.0))
    result = build_scorecard(metrics, minimum_coverage=0.8)
    result["assessment_mode"] = mode
    result["trusted_sample_fraction"] = sample_fraction
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variants = list(CONFIG["variants"])
    audit_count = round(len(SPLITS["train"]) * 0.10)
    audit = set(sorted(SPLITS["train"], key=lambda n: hashlib.sha256(f"audit:{CONFIG['variant_seed']}:{n}".encode()).digest())[:audit_count])
    observations_by_variant = {v: observations(v) for v in variants}
    clean_obs = observations_by_variant["clean"]
    rows = []
    full = {}
    for variant in variants:
        obs = observations_by_variant[variant]
        oracle = reference_metrics(variant)
        limited = reference_metrics(variant, audit)
        cards = {
            "reference_free": scorecard(obs, mode="reference_free"),
            "full_oracle_assisted": scorecard(obs, oracle, mode="full_oracle_assisted"),
            "deterministic_10_percent_annotation_audit": scorecard(obs, limited, mode="deterministic_10_percent_annotation_audit", sample_fraction=0.10),
        }
        baseline = {
            "schema_integrity_detected": any(obs[k] > 0 for k in ("malformed_label_files", "missing_label_files", "unreadable_images")),
            "duplicate_check_detected": obs["duplicate_images_after_first"] > clean_obs["duplicate_images_after_first"],
            "clarity_proxy_detected": obs["low_clarity_images"] > clean_obs["low_clarity_images"] + 0.05 * max(obs["training_images"], 1),
            "annotation_count_detected": obs["box_count"] < 0.98 * clean_obs["box_count"],
            "file_coverage_detected": obs["training_images"] < clean_obs["training_images"],
            "center_distribution_detected": abs((obs["mean_center_x"] or 0) - clean_obs["mean_center_x"]) > 0.02 or abs((obs["mean_center_y"] or 0) - clean_obs["mean_center_y"]) > 0.02,
        }
        full[variant] = {"observations": obs, "oracle_measurements": oracle, "limited_audit_measurements": limited, "scorecards": cards, "conventional_baselines": baseline}
        for mode, card in cards.items():
            rows.append({"variant": variant, "mode": mode, "primary_quality_score": card["primary_quality_score"],
                         "measured_quality_score": card["measured_quality_score"], "score_lower": card["score_bounds"][0],
                         "score_upper": card["score_bounds"][1], "evidence_coverage": card["evidence_coverage"],
                         "readiness_decision": card["readiness_decision"], "warning_count": len(card["warnings"])})
    (OUT / "flightready_assessments.json").write_text(json.dumps({"protocol_version": CONFIG["protocol_version"], "audit_names": sorted(audit), "variants": full}, indent=2), encoding="utf-8")
    with (OUT / "flightready_scorecards.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"variants": len(variants), "scorecards": len(rows), "audit_images": len(audit), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
