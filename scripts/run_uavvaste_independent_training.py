"""Run the preregistered UAVVaste detector experiments and final test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

# Set this before importing Ultralytics so its settings stay inside the workspace.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "ultralytics"))

import torch
from ultralytics import YOLO
from PIL import Image
from ultralytics.data import utils as yolo_data_utils

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
CONFIG = json.loads((EXP / "experiment_config.json").read_text(encoding="utf-8"))
WEIGHTS = EXP / "weights"
RUNS = EXP / "training_runs"
EVALS = EXP / "test_evaluations"
RESULTS = EXP / "model_results"


def read_only_check_image(im_file: str):
    """Ultralytics-compatible integrity check that never rewrites source JPEGs."""
    with Image.open(im_file) as image:
        image.verify()
        width, height = yolo_data_utils.exif_size(image)
        image_format = image.format.lower()
    assert height > 9 and width > 9, f"image size {(height, width)} <10 pixels"
    assert image_format in yolo_data_utils.IMG_FORMATS | {"jpeg2000"}, f"invalid image format {image_format}"
    return "", (height, width)


# Dataset immutability is part of the preregistered protocol. Ultralytics normally
# rewrites JPEGs lacking an EOI marker; validation here is deliberately read-only.
yolo_data_utils.check_image = read_only_check_image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    """Return a portable repository-relative artifact path for public records."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def plans():
    train = CONFIG["training"]
    reduced = CONFIG.get("executed_resource_plan")
    output = []
    for condition in train["primary_conditions"]:
        seeds = reduced["repeat_seeds"] if reduced and condition in reduced["repeat_seed_conditions"] else ([reduced["other_primary_seed"]] if reduced else train["primary_seeds"])
        for seed in seeds:
            output.append(("yolo11n", condition, seed, train["epochs_fixed"], "fixed_epochs"))
    for condition in train["severity_conditions_seed_17"]:
        output.append(("yolo11n", condition, 17, train["epochs_fixed"], "fixed_epochs"))
    for seed in (reduced["fixed_update_seeds"] if reduced else train["fixed_update_seeds"]):
        output.append(("yolo11n", train["fixed_update_condition"], seed, train["fixed_update_epochs"], "fixed_updates"))
    for condition in train["secondary_conditions"]:
        for seed in (reduced["secondary_seeds"] if reduced else train["secondary_seeds"]):
            output.append(("yolov8n", condition, seed, train["epochs_fixed"], "fixed_epochs"))
    return output


def scalar(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def run_one(model_name: str, condition: str, seed: int, epochs: int, schedule: str, *, pilot: bool = False):
    batch = CONFIG["training"]["batch"]
    workers = CONFIG["training"]["workers"]
    tag = f"{model_name}__{condition}__seed{seed}__{schedule}__batch{batch}__workers{workers}"
    output_path = RESULTS / f"{tag}.json"
    if output_path.exists() and not pilot:
        print(f"SKIP {tag}", flush=True)
        return
    run_tag = f"pilot__{tag}" if pilot else tag
    weight_path = WEIGHTS / f"{model_name}.pt"
    data_path = EXP / "dataset" / "prepared" / condition / "dataset.yaml"
    started = time.time()
    model = YOLO(str(weight_path))
    train_results = model.train(
        data=str(data_path), imgsz=CONFIG["training"]["image_size"],
        batch=batch, epochs=1 if pilot else epochs,
        patience=CONFIG["training"]["patience"], optimizer=CONFIG["training"]["optimizer"],
        deterministic=CONFIG["training"]["deterministic"], workers=CONFIG["training"]["workers"],
        device=CONFIG["training"]["device"], seed=seed, project=str(RUNS), name=run_tag,
        exist_ok=True, pretrained=True, cache=False, plots=False, verbose=False,
    )
    save_dir = Path(train_results.save_dir)
    best = save_dir / "weights" / "best.pt"
    test_model = YOLO(str(best))
    test_results = test_model.val(
        data=str(data_path), split="test", imgsz=CONFIG["training"]["image_size"],
        batch=batch, device=CONFIG["training"]["device"],
        workers=CONFIG["training"]["workers"], project=str(EVALS), name=run_tag,
        exist_ok=True, plots=False, save_json=False, verbose=False,
    )
    box = test_results.box
    result = {
        "tag": tag, "pilot": pilot, "protocol_version": CONFIG["protocol_version"],
        "model": model_name, "initial_weights": repository_path(weight_path), "initial_weights_sha256": sha256(weight_path),
        "condition": condition, "seed": seed, "schedule": schedule, "batch": batch, "workers": workers,
        "planned_epochs": 1 if pilot else epochs, "best_checkpoint": repository_path(best),
        "best_checkpoint_sha256": sha256(best), "dataset_yaml": repository_path(data_path),
        "test_split_policy": CONFIG["dataset"]["test_policy"],
        "test_metrics": {
            "precision": scalar(box.mp), "recall": scalar(box.mr),
            "map50": scalar(box.map50), "map50_95": scalar(box.map),
            "per_class_map50_95": [scalar(v) for v in box.maps],
        },
        "elapsed_seconds": time.time() - started,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / (f"pilot__{tag}.json" if pilot else f"{tag}.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"completed": run_tag, "metrics": result["test_metrics"], "seconds": result["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run one epoch of clean seed 17.")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--only-tag", action="append", default=[])
    args = parser.parse_args()
    selected = [("yolo11n", "clean", 17, 1, "fixed_epochs")] if args.pilot else plans()
    if args.only_tag:
        selected = [p for p in selected if any(f"{p[0]}__{p[1]}__seed{p[2]}__{p[4]}" in tag for tag in args.only_tag)]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]
    print(json.dumps({"planned_runs": len(selected), "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}), flush=True)
    for plan in selected:
        run_one(*plan, pilot=args.pilot)


if __name__ == "__main__":
    main()
