"""Build and verify the compact independent-review package and ZIP."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "independent_validation"
PACKAGE_NAME = "FlightReady_Review_Package"
ZIP = EXP / "FlightReady_Review_Package.zip"


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    # Never silently replace a review artifact. The staging directory is temporary
    # and is removed automatically only after a verified archive has been written.
    if ZIP.exists():
        raise FileExistsError(f"Refusing to replace existing review package: {ZIP}")
    files = [
        EXP/"final_reports"/"Results_For_ChatGPT_Review.md",
        EXP/"final_reports"/"FlightReady_Final_Independent_Validation_Report.pdf",
        EXP/"final_reports"/"FlightReady_Final_Independent_Validation_Report.md",
        EXP/"final_reports"/"IEEE_Publication_Readiness_Review.pdf",
        EXP/"final_reports"/"IEEE_Publication_Readiness_Review.md",
        EXP/"final_reports"/"FlightReady_IEEE_Conference_Paper_Draft.md",
        EXP/"final_reports"/"FlightReady_IEEE_Conference_Paper_Draft.pdf",
        EXP/"analysis"/"model_runs.csv", EXP/"analysis"/"condition_summary.csv",
        EXP/"analysis"/"statistical_analysis.json",
        EXP/"quality_results"/"flightready_scorecards.csv",
        EXP/"quality_results"/"flightready_assessments.json",
        EXP/"frozen_methodology.json", EXP/"experiment_config.json", EXP/"variant_manifest.json",
        EXP/"dataset_audit.json", EXP/"REPRODUCIBILITY.md",
        EXP/"protocol_amendment_001.json", EXP/"protocol_amendment_002.json", EXP/"protocol_amendment_003.json", EXP/"protocol_amendment_004.json",
    ] + sorted((EXP/"figures").glob("*.png"))
    with tempfile.TemporaryDirectory(prefix="flightready-review-", dir=EXP) as temporary:
        package = Path(temporary) / PACKAGE_NAME
        package.mkdir()
        for source in files:
            if not source.exists(): raise FileNotFoundError(source)
            folder = "figures" if source.parent.name == "figures" else "evidence" if source.suffix in {".json", ".csv"} else ""
            target = package / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest=[]
        for path in sorted(p for p in package.rglob("*") if p.is_file()):
            manifest.append({"path":path.relative_to(package).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path)})
        (package / "PACKAGE_MANIFEST.json").write_text(json.dumps({"file_count_excluding_manifest":len(manifest),"files":manifest,"excluded":"datasets, caches, pretrained weights, and trained checkpoints"},indent=2),encoding="utf-8")
        with zipfile.ZipFile(ZIP,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for path in sorted(p for p in package.rglob("*") if p.is_file()):
                archive.write(path, Path(PACKAGE_NAME) / path.relative_to(package))
    with zipfile.ZipFile(ZIP) as archive:
        bad=archive.testzip(); names=archive.namelist()
    if bad: raise RuntimeError(f"corrupt ZIP member: {bad}")
    print(json.dumps({"zip":str(ZIP),"zip_bytes":ZIP.stat().st_size,"members":len(names),"sha256":sha256(ZIP)},indent=2))


if __name__ == "__main__":
    main()
