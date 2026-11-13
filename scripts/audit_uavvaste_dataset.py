"""Create a compact provenance, leakage-threat, and immutability audit."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
SOURCE = EXP / "dataset" / "source"
PREPARED = EXP / "dataset" / "prepared"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def group(name):
    stem = Path(name).stem
    stem = re.sub(r"(?:_img)?_?\d+$", "", stem, flags=re.I)
    return stem.lower()


def main():
    coco = json.loads((SOURCE / "annotations" / "annotations.json").read_text(encoding="utf-8"))
    splits = json.loads((SOURCE / "annotations" / "train_val_test_distribution_file.json").read_text(encoding="utf-8"))
    sets = {k: set(v) for k, v in splits.items()}
    groups = {k: {group(v) for v in values} for k, values in splits.items()}
    overlap = {f"{a}_{b}": sorted(groups[a] & groups[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    variants = [p.name for p in PREPARED.iterdir() if p.is_dir()]
    test_issues = []
    for name in splits["test"]:
        source_hash = sha256(SOURCE / "images" / name)
        hashes = {v: sha256(PREPARED / v / "images" / "test" / name) for v in variants}
        if len(set(hashes.values())) != 1 or any(v != source_hash for v in hashes.values()):
            test_issues.append({"file": name, "source_sha256": source_hash, "variant_hashes": hashes})
    output = {
        "dataset": "UAVVaste", "doi": "10.5281/zenodo.8214061", "license": "CC BY 4.0",
        "archive_md5": md5(EXP / "downloads" / "UAVVasteDataset.zip"),
        "annotation_sha256": sha256(SOURCE / "annotations" / "annotations.json"),
        "image_count": len(coco["images"]), "annotation_count": len(coco["annotations"]), "categories": coco["categories"],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "exact_filename_overlap": {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "filename_source_groups": {k: sorted(v) for k, v in groups.items()},
        "source_group_overlap": overlap,
        "leakage_threat": "Official image-level splits share inferred filename source groups; temporally or spatially adjacent frames may cross partitions. No stronger group identifiers were provided, so the official test is retained and this is reported as a threat to validity.",
        "prepared_variants": variants, "test_byte_immutability_issue_count": len(test_issues), "test_byte_immutability_issues": test_issues,
        "weights_sha256": {p.name: sha256(p) for p in (EXP / "weights").glob("*.pt")},
    }
    (EXP / "dataset_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"images": output["image_count"], "annotations": output["annotation_count"], "test_issues": len(test_issues), "group_overlap": {k: len(v) for k,v in overlap.items()}}, indent=2))


if __name__ == "__main__":
    main()
