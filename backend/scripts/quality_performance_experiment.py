"""Fixed-budget mock test of measured training clarity versus detector performance.

No label corruption/repair. Three paired artificial training datasets, one common
general-domain baseline, and unchanged final target/general test sets. The clarity
pass percentage is an explicit experiment criterion, never an overall readiness score.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def mock_dataset(path, seed, domain, clear_fraction=1., external=False):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    ntrain = 48 if domain == 'general' else 32
    count = 24 if external else ntrain+16
    metadata = ['image,split,flight_id,source']
    # Image assignment is paired across quality conditions and fixed before training.
    order = np.random.default_rng(55).permutation(ntrain)
    clear = set(int(i) for i in order[:round(ntrain*clear_fraction)])
    with zipfile.ZipFile(path, 'x') as archive:
        archive.writestr('classes.txt', 'drone\n')
        for i in range(count):
            split = 'test' if external else 'train' if i < ntrain else 'val' if i < ntrain+8 else 'test'
            rng = np.random.default_rng(seed*1000+i)
            image = Image.fromarray(rng.integers(0 if domain == 'general' else 20,
                160 if domain == 'general' else 110, (64,64,3), dtype=np.uint8))
            w = int(rng.integers(12, 35) if domain == 'general' else rng.integers(22, 29))
            h = int(rng.integers(10, 25) if domain == 'general' else rng.integers(14, 21))
            x, y = int(rng.integers(3, 61-w)), int(rng.integers(3, 61-h))
            draw = ImageDraw.Draw(image)
            color = int(rng.integers(190, 256)) if domain == 'general' else 240
            draw.rectangle((x,y,x+w-1,y+h-1), fill=(color,)*3)
            draw.line((x,y+h//2,x+w-1,y+h//2), fill=(255,80,80), width=2)
            if split == 'train' and i not in clear:
                image = image.filter(ImageFilter.GaussianBlur(2.5))
            raw = io.BytesIO(); image.save(raw, format='PNG')
            name = f'images/{split}/{i:03d}.png'
            archive.writestr(name, raw.getvalue())
            archive.writestr(f'labels/{split}/{i:03d}.txt',
                f'0 {(x+w/2)/64} {(y+h/2)/64} {w/64} {h/64}\n')
            metadata.append(f'{name},{split},mock-{seed}-{i},explicitly synthetic {domain} domain')
        archive.writestr('metadata.csv', '\n'.join(metadata)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--general-epochs', type=int, default=100)
    parser.add_argument('--finetune-epochs', type=int, default=60)
    parser.add_argument('--timeout', type=int, default=420)
    args = parser.parse_args()
    root = Path(args.output or '.runtime/quality-performance-'+uuid.uuid4().hex[:8]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ.update(WORKBENCH_ROOT=str(root/'storage'), YOLO_CONFIG_DIR=str(root/'ultralytics'),
        YOLO_OFFLINE='true', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    from fastapi.testclient import TestClient
    from main import app
    from backend.context import DatasetContext, Goal
    from backend.workbench.api import services
    from backend.workbench.datasets import load, image_source
    from scipy.stats import spearmanr
    services.cache_clear()
    started = time.monotonic()

    def post(client, path, body):
        response = client.post('/api/v1'+path, json=body)
        if response.status_code >= 400:
            raise RuntimeError(response.text)
        return response.json()

    def wait(client, job, role):
        last = None
        while time.monotonic()-started < args.timeout:
            response = client.get('/api/v1/jobs/'+job['id']); response.raise_for_status()
            value = response.json(); payload = value['payload']
            stage = payload['stage']
            # Keep progress useful without printing every epoch.
            if stage != last and ('Epoch' not in stage or int(stage.split()[1].split('/')[0]) % 10 == 0):
                print(role+': '+stage, flush=True)
                last = stage
            if payload['status'] == 'completed': return value
            if payload['status'] not in ('queued','running'):
                raise RuntimeError(str(payload)+'\n'+client.get('/api/v1/jobs/'+job['id']+'/logs').text)
            time.sleep(.4)
        post(client, '/jobs/'+job['id']+'/cancel', {})
        raise TimeoutError('Fixed total execution budget reached; no final test tuning performed')

    def upload(client, path):
        with path.open('rb') as incoming:
            response = client.post('/api/v1/datasets/import', files={'file':(path.name,incoming,'application/zip')})
        response.raise_for_status(); return response.json()

    context = DatasetContext(original_text='Detect synthetic drone rectangles; compare measured image clarity against downstream performance.',
        confirmed=True, goal=Goal(task_type='object_detection', required_classes=['drone'], input_modalities=['image']))
    with TestClient(app) as client:
        store, _ = services()
        models, datasets, qualities, ready_ids = {}, {}, {}, {}
        common = dict(batch=8, image_size=64, learning_rate=.01, seed=42, device='cpu',
                      experiment_group='measured-clarity-versus-performance')
        conditions = [('general',1.),('low_clarity',0.),('mixed_clarity',.5),('high_clarity',1.)]
        for role, fraction in conditions:
            domain = 'general' if role == 'general' else 'target'
            dataset = upload(client, mock_dataset(root/(role+'.zip'), seed=311 if role == 'general' else 522,
                domain=domain, clear_fraction=fraction))
            # Mark ALL synthetic conditions artificial; do not pollute natural-source correlations.
            dataset = store.update(dataset['id'], provenance={'artificial':True,
                'experiment':'paired image clarity intervention', 'clear_fraction_requested':fraction})
            datasets[role] = dataset
            split = post(client, '/datasets/'+dataset['id']+'/splits', {'group_column':'flight_id','seed':42})
            ready = wait(client, post(client, '/readiness', {'dataset_id':dataset['id'], 'split_id':split['id'],
                'context':context.model_dump()}), role+' readiness')
            ready_ids[role] = ready['id']
            cert = ready['payload']['result']
            clarity = next(m for m in cert['metrics'] if m['metric_id'] == 'UAV-14')
            frame = load(store, dataset['id']).frame
            train_names = split['payload']['images']['train']
            sharpness = frame.loc[frame.image.isin(train_names),'laplacian_variance']
            assert abs(float(sharpness.mean())-clarity['value']['laplacian_variance']['mean']) < 1e-8
            qualities[role] = {'readiness':cert['overall_status'], 'train_images':len(train_names),
                'pipeline_UAV14':clarity['value'], 'mean_laplacian_variance':float(sharpness.mean()),
                'clarity_rule':'laplacian_variance >= 100; illustrative experiment-only criterion',
                'clarity_pass_percent':100*float((sharpness >= 100).mean()),
                'overall_quality_percent':None, 'dataset_id':dataset['id'], 'split_id':split['id'], 'readiness_id':ready['id']}
            if role != 'general':
                # All annotation bytes and unchanged validation/test pixels must match the high/low paired source.
                with image_source(dataset['payload']) as archive:
                    audit = {name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                        if name.startswith('labels/') or name.startswith('images/val/') or name.startswith('images/test/')}
                if role == 'low_clarity': reference_audit = audit
                else: assert audit == reference_audit, 'Quality conditions changed labels or held-out images'
            config = {**common, 'dataset_id':dataset['id'], 'split_id':split['id'], 'readiness_id':ready['id'],
                      'epochs':args.general_epochs if role == 'general' else args.finetune_epochs,
                      'initialization':'scratch' if role == 'general' else 'pretrained',
                      'model_type':'reference' if role == 'general' else 'controlled'}
            if role != 'general':
                config['weights_path'] = store.get(models['general'])['payload']['artifact']
            job = wait(client, post(client,'/training',config), role+' training')
            models[role] = job['payload']['result']['model_id']
            qualities[role]['training_job_id'] = job['id']
            print(json.dumps({'role':role,'quality':qualities[role]['clarity_pass_percent'],
                'validation':job['payload']['result']['validation']}), flush=True)
        trained = {role:store.get(id,'model')['payload'] for role,id in models.items()}
        assert len({trained[r]['configuration']['initialization_sha256'] for r in models if r != 'general'}) == 1
        # Both final test sets stay unseen until EVERY optimization run has ended.
        evaluations = {}
        for domain, seed in [('target',733),('general',944)]:
            test = upload(client,mock_dataset(root/(domain+'-final-test.zip'),seed,domain,external=True))
            job = wait(client,post(client,'/evaluations',{'model_ids':list(models.values()),
                'dataset_id':test['id'],'device':'cpu','confidence':.25,'iou':.7,'match_iou':.5}), domain+' final test')
            values = [store.get(id,'evaluation') for id in job['payload']['result']['evaluation_ids']]
            assert len({v['payload']['test_sha256'] for v in values}) == 1
            assert all(v['payload']['test_images'] == 24 for v in values)
            evaluations[domain] = {role:value for role,value in zip(models,values)}
        rows = []
        for role in models:
            model = trained[role]
            path = store.directory('jobs',model['job_id'])/'run'/'results.csv'
            with path.open() as incoming: losses = list(csv.DictReader(incoming))
            final_losses = {k:float(v) for k,v in losses[-1].items() if k.startswith(('train/','val/'))}
            rows.append({'condition':role,'quality':qualities[role], 'model_id':models[role],
                'epochs_completed':len(losses),'final_epoch_losses':final_losses,'epoch_metrics_csv':str(path),
                'selected_checkpoint_validation':model['validation'],
                'target_test':evaluations['target'][role]['payload']['metrics'],
                'general_test':evaluations['general'][role]['payload']['metrics']})
        variants = rows[1:]
        association = {}
        for domain in ('target','general'):
            x = [r['quality']['mean_laplacian_variance'] for r in variants]
            y = [r[domain+'_test']['mAP50_95'] for r in variants]
            association[domain] = {'spearman_descriptive':float(spearmanr(x,y).statistic) if len(set(y)) > 1 else None,
                'strictly_improves_with_clarity':all(b>a for a,b in zip(y,y[1:])),
                'high_clarity_minus_low_clarity_mAP50_95_percentage_points':100*(y[-1]-y[0]),
                'high_clarity_minus_general_baseline_mAP50_95_percentage_points':100*(y[-1]-rows[0][domain+'_test']['mAP50_95']),
                'scope':'Three dependent synthetic interventions, one seed; descriptive only, not independent-dataset validation.'}
        summary = {'execution_status':'passed','elapsed_seconds':round(time.monotonic()-started,2),
            'hypothesis':'Higher pipeline-measured training clarity is associated with improved downstream model performance.',
            'overall_readiness_percentage':None, 'overall_readiness_percentage_reason':'FlightReady has no validated composite percentage.',
            'design':{'general_train_images':48,'target_train_images_each':32,'validation_images_each':8,
                'independent_final_test_images_per_domain':24,'general_epochs':args.general_epochs,
                'finetune_epochs_each':args.finetune_epochs,'seed':42,'common_training':common,
                'paired_labels_unchanged':True,'paired_validation_and_internal_test_unchanged':True,
                'identical_finetuning_initialization':True,'exact_near_duplicate_and_group_test_checks':'passed'},
            'rows':rows,'association':association,
            'limitations':['Mock general domain is a broader synthetic generator, not COCO/VisDrone or general-purpose pretrained weights.',
                'The clarity pass percentage is a declared property-specific experiment rule, not an overall quality/readiness percentage.',
                'Image sharpness may also measure noise; this single property is not universal dataset goodness.',
                'One training seed and three paired artificial variants cannot validate a general relationship or causal deployment benefit.',
                'Baseline versus fine-tuned models differ in training duration and domain; only the three fine-tuning conditions have matched adaptation budgets.',
                'Final losses describe the last epoch; test metrics use the validation-selected best checkpoint.',
                'Frozen general-model weights cannot improve from scoring a dataset; fine-tuning is required to change them.',
                'No final test result was used to retune a model, threshold, or quality rule.']}
        (root/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
        with (root/'results.csv').open('w',newline='',encoding='utf-8') as output:
            writer = csv.DictWriter(output,fieldnames=['condition','clarity_pass_percent','mean_laplacian_variance','epochs',
                'target_mAP50','target_mAP50_95','target_precision','target_recall','target_f1',
                'general_mAP50','general_mAP50_95','general_precision','general_recall','general_f1'])
            writer.writeheader()
            for row in rows:
                writer.writerow({'condition':row['condition'],'clarity_pass_percent':row['quality']['clarity_pass_percent'],
                    'mean_laplacian_variance':row['quality']['mean_laplacian_variance'],'epochs':row['epochs_completed'],
                    **{domain+'_'+metric:row[domain+'_test'][metric] for domain in ('target','general')
                       for metric in ('mAP50','mAP50_95','precision','recall','f1')}})
        lines = ['# Measured dataset clarity versus model performance','',
            'Three paired mock target datasets; labels are unchanged. A broader mock baseline is trained first.',
            'Clarity pass percent is an explicit illustrative rule (Laplacian variance >=100), NOT an overall readiness percentage.','',
            '| Training condition | Clarity pass | Target mAP50 | Target mAP50:95 | Target P / R / F1 | General mAP50:95 |',
            '|---|---:|---:|---:|---|---:|']
        def pct(value): return 'undefined' if value is None else f'{100*value:.2f}%'
        for row in rows:
            t,g = row['target_test'],row['general_test']
            lines.append(f"| {row['condition']} | {row['quality']['clarity_pass_percent']:.1f}% | {pct(t['mAP50'])} | {pct(t['mAP50_95'])} | "+
                ' / '.join(pct(t[k]) for k in ('precision','recall','f1'))+f" | {pct(g['mAP50_95'])} |")
        lines += ['', '## Training evidence','',f'Elapsed execution: {summary["elapsed_seconds"]} seconds on CPU.','']
        for row in rows: lines.append(f'- {row["condition"]}: {row["epochs_completed"]} epochs; final losses '+json.dumps(row['final_epoch_losses']))
        lines += ['', '## Descriptive association','',json.dumps(association,indent=2),'','## Limits','']
        lines += ['- '+v for v in summary['limitations']]
        (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps({'elapsed_seconds':summary['elapsed_seconds'],'association':association,
            'results_csv':str(root/'results.csv'),'report':str(root/'report.md')},indent=2),flush=True)


if __name__ == '__main__': main()
