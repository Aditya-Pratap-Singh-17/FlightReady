"""Prepare immutable-split UAVVaste variants for the independent validation.

Images that are not modified are hard-linked to the single extracted source so
the experiment does not duplicate multi-gigabyte data unnecessarily.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
SOURCE = EXP / "dataset" / "source"
PREPARED = EXP / "dataset" / "prepared"
CONFIG = json.loads((EXP / "experiment_config.json").read_text(encoding="utf-8"))


def digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except FileExistsError:
        return
    except OSError:
        if destination.exists():
            return
        shutil.copy2(source, destination)


def selected(names, prevalence, salt):
    count = round(len(names) * prevalence / 100)
    return set(sorted(names, key=lambda n: digest(f"{CONFIG['variant_seed']}:{salt}:{n}"))[:count])


def gaussian(source: Path, destination: Path, name: str, sigma: float):
    if destination.exists():
        return
    with Image.open(source) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    seed = int.from_bytes(digest(f"noise:{CONFIG['variant_seed']}:{name}")[:8], "big")
    rng = np.random.default_rng(seed)
    output = np.clip(array + rng.normal(0, sigma, array.shape), 0, 255).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(destination, "JPEG", quality=95, subsampling=0)


def motion_blur(source: Path, destination: Path, width: int):
    if destination.exists():
        return
    import cv2
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    kernel = np.zeros((width, width), dtype=np.float32)
    kernel[width // 2, :] = 1 / width
    output = cv2.filter2D(rgb, -1, kernel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(destination, "JPEG", quality=95, subsampling=0)


def main():
    annotations = json.loads((SOURCE / "annotations" / "annotations.json").read_text(encoding="utf-8"))
    splits = json.loads((SOURCE / "annotations" / "train_val_test_distribution_file.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in annotations["images"]}
    image_by_name = {row["file_name"]: row for row in annotations["images"]}
    boxes = {name: [] for name in image_by_name}
    for ann in annotations["annotations"]:
        image = by_id[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        boxes[image["file_name"]].append([0, (x + w / 2) / image["width"], (y + h / 2) / image["height"], w / image["width"], h / image["height"]])
    test_fingerprint = hashlib.sha256("\n".join(sorted(splits["test"])).encode()).hexdigest()
    manifests = []
    for name, spec in CONFIG["variants"].items():
        root = PREPARED / name
        affected = selected(splits["train"], spec["prevalence"], spec["family"] + name)
        removed = selected(splits["train"], spec.get("training_images_removed", 0) * 100, "coverage:" + name)
        modification_counts = {"images_modified": 0, "images_removed": len(removed), "boxes_original": 0, "boxes_written": 0,
                               "boxes_removed": 0, "boxes_shifted": 0}
        for split, names in splits.items():
            for image_name in names:
                if split == "train" and image_name in removed:
                    continue
                source_image = SOURCE / "images" / image_name
                destination_image = root / "images" / split / image_name
                if split == "train" and image_name in affected and spec["family"] in ("gaussian_noise", "combined"):
                    gaussian(source_image, destination_image, image_name, spec["sigma_rgb"])
                    modification_counts["images_modified"] += 1
                elif split == "train" and image_name in affected and spec["family"] == "motion_blur":
                    motion_blur(source_image, destination_image, spec["horizontal_kernel_pixels"])
                    modification_counts["images_modified"] += 1
                else:
                    link(source_image, destination_image)
                rows = [list(row) for row in boxes[image_name]]
                modification_counts["boxes_original"] += len(rows)
                if split == "train" and image_name in affected and spec["family"] in ("missing_annotations", "combined"):
                    keep = []
                    for index, row in enumerate(rows):
                        key = digest(f"box:{CONFIG['variant_seed']}:{name}:{image_name}:{index}")
                        if int.from_bytes(key[:8], "big") / 2**64 >= spec["box_removal_fraction_in_affected_images"]:
                            keep.append(row)
                    modification_counts["boxes_removed"] += len(rows) - len(keep)
                    rows = keep
                if split == "train" and image_name in affected and spec["family"] in ("bbox_displacement", "combined"):
                    shift = spec["normalized_center_shift"]
                    for row in rows:
                        row[1] = min(1 - row[3] / 2, max(row[3] / 2, row[1] + shift))
                        row[2] = min(1 - row[4] / 2, max(row[4] / 2, row[2] + shift))
                    modification_counts["boxes_shifted"] += len(rows)
                modification_counts["boxes_written"] += len(rows)
                label_path = root / "labels" / split / (Path(image_name).stem + ".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("".join(f"{int(r[0])} {r[1]:.8f} {r[2]:.8f} {r[3]:.8f} {r[4]:.8f}\n" for r in rows), encoding="utf-8")
        yaml = f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: rubbish\n"
        (root / "dataset.yaml").write_text(yaml, encoding="utf-8")
        actual = {split: sum(1 for _ in (root / "images" / split).glob("*")) for split in splits}
        manifest = {"variant": name, "family": spec["family"], "prevalence_percent": spec["prevalence"],
                    "selection_seed": CONFIG["variant_seed"], "affected_training_images": len(affected),
                    "affected_sha256": hashlib.sha256("\n".join(sorted(affected)).encode()).hexdigest(),
                    "removed_training_images": len(removed), "image_counts": actual,
                    "official_split_counts": {k: len(v) for k, v in splits.items()}, "test_filename_sha256": test_fingerprint,
                    "test_unchanged": actual["test"] == len(splits["test"]), "modification_counts": modification_counts,
                    "rules": spec, "dataset_yaml": (root / "dataset.yaml").relative_to(ROOT).as_posix()}
        (root / "variant_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(manifest)
        print(name, actual, modification_counts, flush=True)
    output = {"protocol_version": CONFIG["protocol_version"], "source_archive_md5": CONFIG["dataset"]["archive_md5"],
              "source_annotation_sha256": file_sha256(SOURCE / "annotations" / "annotations.json"),
              "official_split_sha256": file_sha256(SOURCE / "annotations" / "train_val_test_distribution_file.json"),
              "test_filename_sha256": test_fingerprint, "variants": manifests,
              "wrong_class_status": "not_applicable_single_class"}
    (EXP / "variant_manifest.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
