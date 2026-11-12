"""Export an actual synthetic assessment; never overwrite an existing directory."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from backend.context import DatasetContext,Goal
from backend.ingestion import load_dataset
from backend.readiness_engine import assess
from backend.reports import html_report
from backend.scripts.generate_demos import generate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('.runtime/uav-example-report'))
    args=parser.parse_args()
    with TemporaryDirectory(prefix='flightready-example-') as temporary:
        generate(temporary)
        data=(Path(temporary)/'detection_valid.zip').read_bytes()
    ds=load_dataset('synthetic-demonstration.zip',data)
    try:
        ctx=DatasetContext(original_text='Synthetic software demonstration: detect annotated cars in UAV-like images.',confirmed=True,
            goal=Goal(task_type='object_detection',input_modalities=['image'],required_classes=['car']))
        c=assess(ds,ctx)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'uav-synthetic-report.json').write_text(c.model_dump_json(indent=2),encoding='utf-8')
        (args.output/'uav-synthetic-report.html').write_text(html_report(c),encoding='utf-8')
        print(json.dumps({'status':c.overall_status,'metrics':len(c.metrics),'dataset_sha256':ds.sha256}))
    finally:ds.close()

if __name__=='__main__':main()
