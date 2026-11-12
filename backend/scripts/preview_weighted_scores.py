"""Preview temporary scores on stored assessments without rewriting research evidence."""
import argparse
import csv
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store',required=True)
    parser.add_argument('--output')
    args=parser.parse_args()
    if not (Path(args.store)/'metadata.sqlite3').is_file():
        parser.error('Choose an existing workbench store with metadata.sqlite3')
    from backend.context import Certificate
    from sql.store import Store
    from backend.workbench.datasets import load,subset
    from backend.workbench.splits import get_split
    from metrics.experimental_scoring import calculate
    root=Path(args.output or '.runtime/weighted-score-preview-'+uuid.uuid4().hex[:8]).resolve()
    root.mkdir(parents=True,exist_ok=False)
    store=Store(args.store)
    rows=[]
    for job in reversed(store.list('job')):
        p=job['payload']
        if p['task']!='readiness' or p['status']!='completed': continue
        cert=Certificate.model_validate(p['result'])
        ds=load(store,p['config']['dataset_id'])
        if p['config'].get('split_id'):
            split=get_split(store,p['config']['split_id'],p['config']['dataset_id'])
            ds=subset(ds,split['images']['train'])
        score=calculate(ds,cert.confirmed_context.goal,cert.metrics,cert.findings)
        (root/(job['id']+'.json')).write_text(json.dumps(score,indent=2,allow_nan=False),encoding='utf-8')
        rows.append({'report_id':job['id'],'dataset':cert.dataset['name'],'readiness':cert.overall_status,
            'experimental_score':score['score_percent'],'coverage_percent':score['evidence_coverage_percent'],
            'status':score['status'],'profile_sha256':score.get('profile_sha256')})
    if rows:
        with (root/'scores.csv').open('w',newline='',encoding='utf-8') as incoming:
            writer=csv.DictWriter(incoming,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'preview_only':True,'output':str(root),'rows':rows},indent=2))


if __name__=='__main__': main()
