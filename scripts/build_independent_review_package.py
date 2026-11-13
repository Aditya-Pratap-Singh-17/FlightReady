"""Build and verify the compact independent-review package and ZIP."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
PACKAGE = EXP / "review_package"
ZIP = EXP / "FlightReady_Review_Package.zip"


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    if PACKAGE.exists():
        for path in sorted(PACKAGE.rglob("*"), reverse=True):
            if path.is_file(): path.unlink()
            elif path.is_dir(): path.rmdir()
    PACKAGE.mkdir(parents=True,exist_ok=True)
    files = [
        EXP/"reports"/"Results_For_Independent_Review.md",
        EXP/"reports"/"FlightReady_Independent_Validation_Report.pdf",
        EXP/"reports"/"FlightReady_Independent_Validation_Report.md",
        EXP/"reports"/"IEEE_Publication_Readiness_Assessment.pdf",
        EXP/"reports"/"IEEE_Publication_Readiness_Assessment.md",
        EXP/"reports"/"FlightReady_IEEE_Conference_Paper_Draft.md",
        EXP/"reports"/"FlightReady_IEEE_Conference_Paper_Draft.pdf",
        EXP/"analysis"/"model_runs.csv", EXP/"analysis"/"condition_summary.csv",
        EXP/"analysis"/"statistical_analysis.json",
        EXP/"quality_results"/"flightready_scorecards.csv",
        EXP/"quality_results"/"flightready_assessments.json",
        EXP/"frozen_methodology.json", EXP/"experiment_config.json", EXP/"variant_manifest.json",
        EXP/"dataset_audit.json", EXP/"REPRODUCIBILITY.md",
        EXP/"protocol_amendment_001.json", EXP/"protocol_amendment_002.json", EXP/"protocol_amendment_003.json", EXP/"protocol_amendment_004.json",
    ] + sorted((EXP/"figures").glob("*.png"))
    for source in files:
        if not source.exists(): raise FileNotFoundError(source)
        folder = "figures" if source.parent.name == "figures" else "evidence" if source.suffix in {".json",".csv"} else ""
        target = PACKAGE/folder/source.name
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    manifest=[]
    for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file()):
        manifest.append({"path":path.relative_to(PACKAGE).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path)})
    (PACKAGE/"PACKAGE_MANIFEST.json").write_text(json.dumps({"file_count_excluding_manifest":len(manifest),"files":manifest,"excluded":"datasets, caches, pretrained weights, and trained checkpoints"},indent=2),encoding="utf-8")
    if ZIP.exists(): ZIP.unlink()
    with zipfile.ZipFile(ZIP,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file()): archive.write(path,path.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(ZIP) as archive:
        bad=archive.testzip(); names=archive.namelist()
    if bad: raise RuntimeError(f"corrupt ZIP member: {bad}")
    print(json.dumps({"zip":str(ZIP),"zip_bytes":ZIP.stat().st_size,"members":len(names),"sha256":sha256(ZIP)},indent=2))


if __name__ == "__main__":
    main()
