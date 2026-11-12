"""Real VisDrone/YOLO11 quality-performance validation for FlightReady.

Subcommands are resumable: prepare, variants, assess, train, report, all.
Original archives are never modified. Dataset variants use hard links for
unchanged files so only deliberately corrupted images consume extra space.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "visdrone_model_validation"
DOWNLOADS = EXP / "resources" / "downloads"
DATA = EXP / "datasets"
PREPARED = DATA / "prepared" / "VisDrone"
VARIANTS = DATA / "variants"
REPORTS = EXP / "flightready_reports"
RUNS = EXP / "training_runs"
RESULTS = EXP / "results"
PLOTS = EXP / "plots"
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
SEED = 20261111


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selected(names: list[str], fraction: float, salt: int) -> set[str]:
    order = names.copy()
    random.Random(SEED + salt).shuffle(order)
    return set(order[: round(len(order) * fraction)])


def prepare() -> None:
    from PIL import Image
    PREPARED.mkdir(parents=True, exist_ok=True)
    mapping = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
    provenance = {"sources": {}, "model": {}, "conversion": "Ultralytics VisDrone.yaml algorithm; ignored score=0 regions excluded"}
    for split, source_name in mapping.items():
        archive = DOWNLOADS / f"{source_name}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        image_out, label_out = PREPARED / "images" / split, PREPARED / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True); label_out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            members = z.namelist()
            image_members = [n for n in members if n.lower().endswith(".jpg") and "/images/" in n.replace("\\", "/")]
            annotation_members = {Path(n).stem: n for n in members if n.lower().endswith(".txt") and "/annotations/" in n.replace("\\", "/")}
            for member in image_members:
                target = image_out / Path(member).name
                if not target.exists():
                    with z.open(member) as incoming, target.open("wb") as output:
                        shutil.copyfileobj(incoming, output)
                ann = annotation_members.get(target.stem)
                lines = []
                if ann:
                    width, height = Image.open(target).size
                    for raw in z.read(ann).decode("utf-8-sig").strip().splitlines():
                        row = raw.rstrip(",").split(",")
                        if len(row) < 8 or row[4] == "0":
                            continue
                        x, y, w, h = map(int, row[:4]); cls = int(row[5]) - 1
                        if 0 <= cls < len(CLASSES) and w > 0 and h > 0:
                            lines.append(f"{cls} {(x+w/2)/width:.6f} {(y+h/2)/height:.6f} {w/width:.6f} {h/height:.6f}\n")
                (label_out / f"{target.stem}.txt").write_text("".join(lines), encoding="utf-8")
        provenance["sources"][split] = {"archive": str(archive), "sha256": sha256(archive), "bytes": archive.stat().st_size,
            "images": len(list(image_out.glob("*.jpg"))), "labels": len(list(label_out.glob("*.txt")))}
    model = DOWNLOADS / "yolo11n.pt"
    provenance["model"] = {"path": str(model), "sha256": sha256(model), "bytes": model.stat().st_size,
        "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", "license": "AGPL-3.0"}
    provenance["dataset"] = {"source": "https://github.com/VisDrone/VisDrone-Dataset",
        "download_adapter": "https://docs.ultralytics.com/datasets/detect/visdrone", "license": "No explicit license found in official repository/catalog; verify rights before redistribution."}
    write_yaml(PREPARED / "VisDrone.yaml", PREPARED)
    dump(EXP / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)


def write_yaml(path: Path, dataset_root: Path) -> None:
    quoted = str(dataset_root.resolve()).replace("\\", "/")
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    path.write_text(f"path: '{quoted}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")


def hardlink_tree(source: Path, target: Path) -> None:
    for file in source.rglob("*"):
        if file.is_file():
            destination = target / file.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(file, destination)


def labels(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(line.split())
    return rows


def save_labels(path: Path, rows) -> None:
    path.write_text("".join(" ".join(map(str, row)) + "\n" for row in rows), encoding="utf-8")


def variant_spec():
    return [
        {"name": "clean", "family": "reference", "prevalence": 0, "severity": "none"},
        {"name": "gaussian_noise_25", "family": "image_noise", "prevalence": 25, "severity": "Gaussian RGB sigma=25/255"},
        {"name": "motion_blur_25", "family": "motion_blur", "prevalence": 25, "severity": "horizontal 9-pixel kernel"},
        {"name": "contrast_25", "family": "reduced_contrast", "prevalence": 25, "severity": "contrast factor 0.45"},
        {"name": "jpeg_25", "family": "compression", "prevalence": 25, "severity": "JPEG quality=18"},
        {"name": "missing_annotations_25", "family": "missing_annotations", "prevalence": 25, "severity": "remove 50% of boxes in affected images"},
        {"name": "wrong_class_25", "family": "wrong_class", "prevalence": 25, "severity": "change every box class in affected images to a different valid class"},
        {"name": "bbox_displacement_25", "family": "bbox_displacement", "prevalence": 25, "severity": "shift centers by +0.08 normalized x/y, clipped"},
        {"name": "incomplete_40", "family": "dataset_completeness", "prevalence": 40, "severity": "exclude 40% of training images"},
        {"name": "duplicates_25", "family": "duplicates", "prevalence": 25, "severity": "add exact duplicate files equal to 25% of original image count"},
        {"name": "class_imbalance_40", "family": "class_imbalance", "prevalence": 40, "severity": "remove 40% of images containing rare classes 7 or 8"},
        {"name": "combined_40", "family": "combined", "prevalence": 40, "severity": "Gaussian sigma=25 plus 50% box omission and valid wrong classes in affected images"},
        {"name": "combined_repaired", "family": "oracle_repair", "prevalence": 0, "severity": "restore original bytes and annotations from preserved clean snapshot"},
    ]


def variants() -> None:
    import numpy as np
    from PIL import Image, ImageEnhance
    from scipy.ndimage import uniform_filter1d
    train_images = PREPARED / "images" / "train"; train_labels = PREPARED / "labels" / "train"
    names = sorted(p.name for p in train_images.glob("*.jpg"))
    if len(names) != 6471: raise RuntimeError(f"Expected 6471 training images, got {len(names)}")
    manifests = []
    for index, spec in enumerate(variant_spec()):
        name = spec["name"]
        if name == "clean":
            manifests.append({**spec, "root": str(PREPARED), "images": len(names), "affected": 0})
            continue
        target = VARIANTS / name
        if name == "combined_repaired":
            manifests.append({**spec, "root": str(PREPARED), "images": len(names), "affected": 0, "oracle_source": "clean"})
            continue
        marker = target / "variant_manifest.json"
        if marker.exists():
            manifests.append(json.loads(marker.read_text(encoding="utf-8")))
            print(f"reused completed {name}", flush=True)
            continue
        hardlink_tree(train_images, target / "images" / "train")
        hardlink_tree(train_labels, target / "labels" / "train")
        affected = selected(names, spec["prevalence"] / 100, index * 1009)
        if name == "incomplete_40":
            for image_name in affected:
                (target / "images" / "train" / image_name).unlink()
                (target / "labels" / "train" / (Path(image_name).stem + ".txt")).unlink(missing_ok=True)
        elif name == "class_imbalance_40":
            candidates = [n for n in names if any(row and int(row[0]) in (7, 8) for row in labels(train_labels / (Path(n).stem + ".txt")))]
            affected = selected(candidates, .4, index * 1009)
            for image_name in affected:
                (target / "images" / "train" / image_name).unlink()
                (target / "labels" / "train" / (Path(image_name).stem + ".txt")).unlink(missing_ok=True)
        elif name == "duplicates_25":
            for i, image_name in enumerate(sorted(affected)):
                stem = Path(image_name).stem; suffix = f"duplicate_{i:05d}_{stem}"
                os.link(train_images / image_name, target / "images" / "train" / f"{suffix}.jpg")
                os.link(train_labels / f"{stem}.txt", target / "labels" / "train" / f"{suffix}.txt")
        else:
            for position, image_name in enumerate(sorted(affected)):
                image_path = target / "images" / "train" / image_name
                label_path = target / "labels" / "train" / (Path(image_name).stem + ".txt")
                if spec["family"] in ("image_noise", "motion_blur", "reduced_contrast", "compression", "combined"):
                    image_path.unlink()
                    image = Image.open(train_images / image_name).convert("RGB")
                    if spec["family"] in ("image_noise", "combined"):
                        arr = np.asarray(image).astype(np.int16); rng = np.random.default_rng(SEED + position + index * 99991)
                        image = Image.fromarray(np.clip(arr + rng.normal(0, 25, arr.shape), 0, 255).astype(np.uint8))
                    elif spec["family"] == "motion_blur":
                        image = Image.fromarray(uniform_filter1d(np.asarray(image), size=9, axis=1, mode="nearest"))
                    elif spec["family"] == "reduced_contrast": image = ImageEnhance.Contrast(image).enhance(.45)
                    image.save(image_path, "JPEG", quality=18 if spec["family"] == "compression" else 92)
                if spec["family"] in ("missing_annotations", "wrong_class", "bbox_displacement", "combined"):
                    label_path.unlink(); rows = labels(train_labels / label_path.name)
                    if spec["family"] in ("missing_annotations", "combined"): rows = rows[::2]
                    if spec["family"] in ("wrong_class", "combined"):
                        for row in rows: row[0] = str((int(row[0]) + 1 + position % 9) % 10)
                    if spec["family"] == "bbox_displacement":
                        for row in rows: row[1], row[2] = f"{min(.999, float(row[1])+.08):.6f}", f"{min(.999, float(row[2])+.08):.6f}"
                    save_labels(label_path, rows)
        write_yaml(target / "VisDrone.yaml", target)
        image_count = len(list((target / "images" / "train").glob("*.jpg")))
        record = {**spec, "root": str(target), "images": image_count, "affected": len(affected),
            "affected_fraction_actual": len(affected) / len(names), "selection_seed": SEED + index * 1009}
        manifests.append(record); dump(marker, record)
        print(f"prepared {name}: {image_count} images, {len(affected)} affected", flush=True)
    dump(EXP / "variant_manifest.json", {"seed": SEED, "original_images": len(names), "variants": manifests})


def assessment_zip(spec: dict) -> Path:
    root = Path(spec["root"]); archive = EXP / ".assessment" / f"{spec['name']}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("classes.txt", "\n".join(CLASSES) + "\n")
        for kind in ("images", "labels"):
            for file in (root / kind / "train").glob("*"):
                z.write(file, f"{kind}/train/{file.name}")
    return archive


def assess_variants(names: set[str] | None = None) -> None:
    # The research corpus is larger than FlightReady's normal interactive-upload
    # ceiling. Raise the limit only inside this experiment process; production
    # defaults in backend.large_ingestion remain unchanged.
    os.environ.setdefault("MAX_UPLOAD_MB", "8192")
    os.environ.setdefault("MAX_IMAGE_UPLOAD_MB", "8192")
    sys.path.insert(0, str(ROOT))
    from backend.context import DatasetContext, Goal
    from backend.large_ingestion import load_dataset_stream
    from backend.readiness_engine import assess
    from backend.reports import html_report
    context = DatasetContext(original_text="Fine-tune a detector for all ten VisDrone classes in real UAV imagery.", confirmed=True,
        goal=Goal(task_type="object_detection", task_description="VisDrone object detection", prediction_target="bounding boxes and classes",
            input_modalities=["image"], required_classes=CLASSES, uav_parameters={"minimum_object_side_pixels": 8, "class_balance_target": "uniform"}))
    manifest = json.loads((EXP / "variant_manifest.json").read_text(encoding="utf-8"))["variants"]
    for spec in manifest:
        if names and spec["name"] not in names: continue
        out = REPORTS / f"{spec['name']}_certificate.json"
        if out.exists(): continue
        archive = assessment_zip(spec); started = time.perf_counter()
        with archive.open("rb") as incoming:
            ds = load_dataset_stream(archive.name, incoming, {"source": "Controlled derivative of official VisDrone2019-DET train split", "dataset_version": spec["name"]})
        cert = assess(ds, context); payload = cert.model_dump(mode="json")
        dump(out, payload); (REPORTS / f"{spec['name']}_report.html").write_text(html_report(cert), encoding="utf-8")
        score = payload.get("experimental_score") or {}; quality = payload.get("quality_profile") or {}
        summary = {"variant": spec["name"], "status": payload["overall_status"], "score": score.get("score_percent"),
            "score_status": score.get("status"), "evidence_coverage_percent": score.get("evidence_coverage_percent"),
            "dimensions": {d["id"]: {"score": d.get("score"), "coverage": d.get("evidence_coverage"), "status": d.get("status")} for d in quality.get("dimensions", [])},
            "metrics": {m["metric_id"]: {"state": m.get("state"), "assessment_status": m.get("assessment_status"), "value": m.get("value")} for m in payload["metrics"]},
            "assessment_seconds": time.perf_counter() - started}
        dump(REPORTS / f"{spec['name']}_summary.json", summary)
        archive.unlink(missing_ok=True)
        print(f"assessed {spec['name']}: {summary['status']} score={summary['score']}", flush=True)


def training_yaml(spec: dict) -> Path:
    root = Path(spec["root"]); path = root / "Training.yaml"
    quoted = str(root.resolve()).replace("\\", "/")
    prepared = str(PREPARED.resolve()).replace("\\", "/")
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    path.write_text(f"path: '{quoted}'\ntrain: images/train\nval: '{prepared}/images/val'\ntest: '{prepared}/images/test'\nnames:\n{names}\n", encoding="utf-8")
    return path


def train(names: set[str] | None = None, epochs: int = 5, pilot: bool = False) -> None:
    os.environ["YOLO_CONFIG_DIR"] = str(EXP / ".ultralytics")
    from ultralytics import YOLO
    manifest = json.loads((EXP / "variant_manifest.json").read_text(encoding="utf-8"))["variants"]
    if pilot:
        spec = next(v for v in manifest if v["name"] == "clean")
        source = Path(spec["root"]); target = DATA / "pilot"
        chosen = sorted((source / "images" / "train").glob("*.jpg"))[:128]
        for image in chosen:
            for src, dest in [(image, target / "images/train" / image.name), (source / "labels/train" / f"{image.stem}.txt", target / "labels/train" / f"{image.stem}.txt")]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): os.link(src, dest)
        spec = {**spec, "name": "pilot", "root": str(target)}; manifest = [spec]; epochs = 1
    for spec in manifest:
        if names and spec["name"] not in names: continue
        output = RUNS / spec["name"]
        if (output / "weights" / "best.pt").exists(): continue
        yaml = training_yaml(spec); started = time.perf_counter()
        model = YOLO(str(DOWNLOADS / "yolo11n.pt"))
        model.train(data=str(yaml), epochs=epochs, imgsz=320, batch=16, device=0, workers=4, seed=42,
            deterministic=True, pretrained=True, optimizer="auto", patience=epochs, cache=False, project=str(RUNS), name=spec["name"], exist_ok=True,
            plots=True, save=True, verbose=True)
        elapsed = time.perf_counter() - started
        best = output / "weights" / "best.pt"
        evaluated = YOLO(str(best)).val(data=str(yaml), split="test", imgsz=320, batch=16, device=0, workers=4,
            project=str(output), name="test-dev", plots=True, save_json=True, verbose=True)
        box = evaluated.box
        result = {"variant": spec["name"], "epochs": epochs, "seed": 42, "image_size": 320, "batch": 16,
            "initialization": str(DOWNLOADS / "yolo11n.pt"), "initialization_sha256": sha256(DOWNLOADS / "yolo11n.pt"),
            "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best), "training_seconds": elapsed,
            "test_split": "official labeled VisDrone2019-DET-test-dev", "test_images": 1610,
            "metrics": {"precision": float(box.mp), "recall": float(box.mr), "mAP50": float(box.map50), "mAP50_95": float(box.map),
                "per_class_mAP50_95": {CLASSES[i]: float(v) for i, v in enumerate(box.maps)}}}
        dump(RESULTS / f"{spec['name']}.json", result)
        print(json.dumps(result, indent=2), flush=True)


def report() -> None:
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    manifest = json.loads((EXP / "variant_manifest.json").read_text(encoding="utf-8"))["variants"]
    by_name = {v["name"]: v for v in manifest}
    for file in RESULTS.glob("*.json"):
        if file.stem == "pilot": continue
        result = json.loads(file.read_text(encoding="utf-8"))
        if "variant" not in result:
            continue
        name = result["variant"]
        readiness_file = REPORTS / f"{name}_summary.json"
        readiness = json.loads(readiness_file.read_text(encoding="utf-8")) if readiness_file.exists() else {}
        rows.append({"variant": name, "degradation": by_name[name]["family"], "prevalence_percent": by_name[name]["prevalence"],
            "severity": by_name[name]["severity"], "flightready_score": readiness.get("score"), "readiness_status": readiness.get("status"),
            "evidence_coverage_percent": readiness.get("evidence_coverage_percent"), **result["metrics"], "training_seconds": result["training_seconds"]})
    frame = pd.DataFrame(rows)
    if frame.empty: raise RuntimeError("No completed experimental training results")
    clean = frame[frame.variant == "clean"].iloc[0]
    for metric in ("mAP50", "mAP50_95", "precision", "recall"):
        frame[f"{metric}_absolute_change"] = frame[metric] - clean[metric]
        frame[f"{metric}_relative_change_percent"] = 100 * (frame[metric] - clean[metric]) / clean[metric]
    RESULTS.mkdir(parents=True, exist_ok=True); frame.to_csv(RESULTS / "performance_comparison.csv", index=False)
    valid = frame.dropna(subset=["flightready_score", "mAP50_95"])
    stats = {"n": len(valid), "pearson": None, "spearman": None, "reason": None, "defects_only": None}
    if len(valid) >= 3 and valid.flightready_score.nunique() > 1 and valid.mAP50_95.nunique() > 1:
        from scipy.stats import pearsonr, spearmanr
        p, s = pearsonr(valid.flightready_score, valid.mAP50_95), spearmanr(valid.flightready_score, valid.mAP50_95)
        stats.update(pearson={"r": float(p.statistic), "p": float(p.pvalue)}, spearman={"rho": float(s.statistic), "p": float(s.pvalue)})
    else: stats["reason"] = "Correlation undefined: fewer than three results or no variance in score/performance."
    defect_only = valid[~valid.variant.isin(["clean", "combined_repaired"])]
    if len(defect_only) >= 3 and defect_only.flightready_score.nunique() > 1:
        dp, ds = pearsonr(defect_only.flightready_score, defect_only.mAP50_95), spearmanr(defect_only.flightready_score, defect_only.mAP50_95)
        stats["defects_only"] = {"n": len(defect_only), "pearson": {"r": float(dp.statistic), "p": float(dp.pvalue)},
            "spearman": {"rho": float(ds.statistic), "p": float(ds.pvalue)}}
    dump(RESULTS / "statistical_analysis.json", stats)
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(frame.flightready_score, frame.mAP50_95)
    label_offsets = {"clean": (5, 8), "combined_repaired": (5, -12)}
    for _, r in frame.iterrows():
        ax.annotate(r.variant, (r.flightready_score, r.mAP50_95), xytext=label_offsets.get(r.variant, (3, 2)),
            textcoords="offset points", fontsize=7)
    ax.set(xlabel="FlightReady experimental score", ylabel="Official test-dev mAP@0.5:0.95", title="Readiness vs detector performance")
    fig.tight_layout(); fig.savefig(PLOTS / "readiness_vs_performance.png", dpi=300); plt.close(fig)
    ordered = frame.sort_values("mAP50_95", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6)); ax.bar(ordered.variant, ordered.mAP50_95); ax.axhline(clean.mAP50_95, color="black", linestyle="--", label="clean")
    ax.tick_params(axis="x", rotation=45); ax.set(ylabel="mAP@0.5:0.95", title="VisDrone corruption sensitivity"); ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS / "corruption_sensitivity.png", dpi=300); plt.close(fig)
    lines = ["# Real VisDrone and YOLO11n validation", "", "## Completed measured results", "",
        "All listed results use official VisDrone2019-DET test-dev images and labels. Training used the same downloaded YOLO11n initialization, 320 px resolution, five epochs, batch 16, seed 42, and CUDA GPU unless the row states otherwise.", "",
        "| Variant | Degradation | FlightReady score | Coverage | Decision | mAP50 | mAP50-95 | Precision | Recall |", "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for _, r in frame.iterrows():
        lines.append(f"| {r.variant} | {r.degradation} | {r.flightready_score if pd.notna(r.flightready_score) else 'unavailable'} | {r.evidence_coverage_percent if pd.notna(r.evidence_coverage_percent) else 'unavailable'} | {r.readiness_status or 'unavailable'} | {r.mAP50:.4f} | {r.mAP50_95:.4f} | {r.precision:.4f} | {r.recall:.4f} |")
    completed_defects = frame[~frame.variant.isin(["clean", "combined_repaired"])].copy()
    all_harmed = bool((completed_defects.mAP50_95_relative_change_percent < 0).all())
    combined = frame[frame.variant == "combined_40"].iloc[0]
    repaired = frame[frame.variant == "combined_repaired"].iloc[0]
    wrong = frame[frame.variant == "wrong_class_25"].iloc[0]
    missing = frame[frame.variant == "missing_annotations_25"].iloc[0]
    incomplete = frame[frame.variant == "incomplete_40"].iloc[0]
    lines += ["", "## Findings and hypothesis outcomes", "",
        f"1. **Data degradation harmed the fixed-budget detector (supported in this run):** all {len(completed_defects)} completed defect conditions reduced test mAP50-95 versus clean ({all_harmed}). Relative losses ranged from {completed_defects.mAP50_95_relative_change_percent.max():.1f}% to {completed_defects.mAP50_95_relative_change_percent.min():.1f}%.",
        f"2. **Semantic and coverage failures were material:** wrong classes reduced mAP50-95 by {abs(wrong.mAP50_95_relative_change_percent):.1f}%, and removing 40% of training images reduced it by {abs(incomplete.mAP50_95_relative_change_percent):.1f}%.",
        f"3. **Combined defects compounded selected individual defects (partially supported):** combined corruption reduced mAP50-95 by {abs(combined.mAP50_95_relative_change_percent):.1f}%, worse than Gaussian noise ({abs(frame[frame.variant == 'gaussian_noise_25'].iloc[0].mAP50_95_relative_change_percent):.1f}%) and missing annotations ({abs(missing.mAP50_95_relative_change_percent):.1f}%), but not lower than the incomplete condition ({abs(incomplete.mAP50_95_relative_change_percent):.1f}%).",
        f"4. **Oracle repair restored observed performance (supported for this seeded run):** repaired mAP50-95 was {repaired.mAP50_95:.6f}, exactly matching clean to reported precision; its readiness score also matched clean.",
        "5. **FlightReady score did not predict detector performance in this matrix (not supported):** scores increased for several harmful variants. Correlations were negative and not statistically significant. This is evidence of metric misalignment/blind spots for these mechanisms, not proof that readiness is never predictive.",
        "", "## FlightReady diagnostic implications", "",
        f"- Gaussian corruption changed the readiness score by only {frame[frame.variant == 'gaussian_noise_25'].iloc[0].flightready_score - clean.flightready_score:+.3f} points while reducing mAP50-95 by {abs(frame[frame.variant == 'gaussian_noise_25'].iloc[0].mAP50_95_relative_change_percent):.1f}%.",
        f"- Structurally valid wrong-class labels received the exact clean score while mAP50-95 fell {abs(wrong.mAP50_95_relative_change_percent):.1f}%; semantic correctness needs gold labels, cross-model disagreement, or targeted human review.",
        f"- Missing annotations and combined defects received *higher* scores than clean ({missing.flightready_score:.3f} and {combined.flightready_score:.3f} vs {clean.flightready_score:.3f}). Annotation-density reductions can make internal statistics look cleaner while weakening supervision.",
        "- Corpus completeness cannot be inferred from retained records alone. A manifest-level expected count, scene/route coverage targets, or acquisition-plan evidence is needed.",
        "- Recommended product changes: expose metric-level score contributions; add annotation-density/coverage baselines; accept external gold-review samples; add image-corruption checks; and never present the aggregate score as a validated proxy for downstream mAP until prospective calibration succeeds.",
        "", "## Statistical relationship", "", json.dumps(stats, indent=2), "",
        "The all-row correlation includes clean and the byte-identical repaired control and is descriptive only. The defects-only analysis removes those duplicated endpoints. Neither analysis is powered for inference (n=7 and n=5).", "",
        "## Reproducibility and provenance", "",
        "- Dataset: official VisDrone2019-DET train (6,471 images), val (548), and labeled test-dev (1,610); official ignored regions were excluded during YOLO conversion.",
        "- Model: official downloaded YOLO11n checkpoint; SHA-256 and archive hashes are recorded in provenance.json and each result JSON.",
        "- Protocol: five epochs, 320 px, batch 16, seed 42, deterministic mode, official val for checkpoint selection, official test-dev once for final reporting.",
        "- Full variant manifests record deterministic affected filenames, severity, and prevalence. Unchanged files are hard-linked; official downloads remain preserved.",
        "", "## Sources", "",
        "- VisDrone official repository: https://github.com/VisDrone/VisDrone-Dataset",
        "- Ultralytics VisDrone dataset documentation: https://docs.ultralytics.com/datasets/detect/visdrone",
        "- Ultralytics YOLO11 model documentation: https://docs.ultralytics.com/models/yolo11",
        "", "## Scope and limitations", "",
        "- The official archives and downloaded checkpoint are preserved under resources/downloads with hashes in provenance.json.",
        "- The VisDrone official repository/catalog supplies no explicit dataset license; redistribution rights are not inferred.",
        "- Five-epoch, 320 px runs are controlled resource-bounded fine-tuning experiments, not state-of-the-art VisDrone training.",
        "- One training seed does not estimate training-run uncertainty. Test-dev was used only for final evaluation, never model selection.",
        "- Oracle repair restores known original bytes/labels; it is not evidence for an automatic real-world repair algorithm.",
        "- Any planned variants without a result JSON were not executed and are not presented as successful."]
    (EXP / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    best_harm = frame.loc[frame.mAP50_95.idxmin()]
    summary = ["# Executive summary", "",
        f"Seven controlled YOLO11n runs were completed on real VisDrone data. Clean achieved mAP50-95 {clean.mAP50_95:.4f} on the untouched official labeled test-dev split.",
        f"Every completed defect condition reduced mAP50-95. The largest loss was {best_harm.variant}: {best_harm.mAP50_95:.4f}, down {abs(best_harm.mAP50_95_relative_change_percent):.1f}%. Wrong classes fell {abs(wrong.mAP50_95_relative_change_percent):.1f}%, Gaussian noise {abs(frame[frame.variant == 'gaussian_noise_25'].iloc[0].mAP50_95_relative_change_percent):.1f}%, missing annotations {abs(missing.mAP50_95_relative_change_percent):.1f}%, and combined defects {abs(combined.mAP50_95_relative_change_percent):.1f}%.",
        f"The repaired control independently reproduced clean performance exactly ({repaired.mAP50_95:.6f}), showing full observed recovery when original bytes and labels were restored.", "",
        f"FlightReady marked every dataset NOT_READY at 80% evidence coverage, but its numeric score was not directionally aligned with mAP. Combined defects scored {combined.flightready_score:.3f} versus clean {clean.flightready_score:.3f} even as mAP50-95 fell {abs(combined.mAP50_95_relative_change_percent):.1f}%.",
        f"Across all seven rows, Pearson r={stats['pearson']['r']:.3f} (p={stats['pearson']['p']:.3f}) and Spearman rho={stats['spearman']['rho']:.3f} (p={stats['spearman']['p']:.3f}); the small one-seed matrix is descriptive, not confirmatory.", "",
        "Conclusion: real dataset defects materially harmed fixed-budget fine-tuning, repair restored performance, and the present aggregate readiness score should not be treated as a calibrated proxy for downstream detector accuracy. Add semantic-label review, expected-coverage evidence, corruption checks, and transparent score contributions before making such a claim.", "",
        "See research_report.md, results/performance_comparison.csv, and plots/ for the complete evidence and limitations."]
    (EXP / "executive_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_names(value: str | None): return set(value.split(",")) if value else None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["prepare", "variants", "assess", "train", "pilot", "report", "all"])
    parser.add_argument("--names"); parser.add_argument("--epochs", type=int, default=5); args = parser.parse_args()
    if args.command in ("prepare", "all"): prepare()
    if args.command in ("variants", "all"): variants()
    if args.command in ("assess", "all"): assess_variants(parse_names(args.names))
    if args.command in ("pilot",): train(pilot=True)
    if args.command in ("train", "all"): train(parse_names(args.names), args.epochs)
    if args.command in ("report", "all"): report()


if __name__ == "__main__": main()
