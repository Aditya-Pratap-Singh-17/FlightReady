"""Short, reproducible synthetic baseline/raw-label/corrected-label experiment.

Uses the real API, readiness worker, YOLO training and independent evaluation.
Corrections use generator ground truth, NOT automatic label repair by FlightReady.
The supplied baseline must be an existing trusted workbench model record.
"""
import argparse
import copy
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


def fixture(path, seed, external=False):
    import numpy as np
    from PIL import Image, ImageDraw
    metadata = ['image,split,flight_id,source']
    with zipfile.ZipFile(path, 'x') as archive:
        archive.writestr('classes.txt', 'drone\n')
        for index in range(20 if external else 48):
            split = 'test' if external else 'train' if index < 32 else 'val' if index < 40 else 'test'
            rng = np.random.default_rng(seed * 1000 + index)
            image = Image.fromarray(rng.integers(0, 150, (64, 64, 3), dtype=np.uint8))
            w, h = int(rng.integers(18, 29)), int(rng.integers(12, 21))
            x, y = int(rng.integers(4, 60-w)), int(rng.integers(4, 60-h))
            draw = ImageDraw.Draw(image)
            draw.rectangle((x, y, x+w-1, y+h-1), fill=(230, 230, 230))
            draw.line((x, y+h//2, x+w-1, y+h//2), fill=(255, 80, 80), width=2)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            name = f'images/{split}/{index:03d}.png'
            archive.writestr(name, buffer.getvalue())
            archive.writestr(f'labels/{split}/{index:03d}.txt',
                             f'0 {(x+w/2)/64} {(y+h/2)/64} {w/64} {h/64}\n')
            metadata.append(f'{name},{split},synthetic-{seed}-{index},explicit mock drawing')
        archive.writestr('metadata.csv', '\n'.join(metadata)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-store', required=True)
    parser.add_argument('--baseline-model', required=True)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--timeout', type=int, default=480, help='Total experiment seconds, including training')
    parser.add_argument('--output')
    args = parser.parse_args()
    started = time.monotonic()
    directory = Path(args.output or '.runtime/quick-accuracy-'+uuid.uuid4().hex[:8]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.environ['WORKBENCH_ROOT'] = str(directory/'storage')
    os.environ['YOLO_CONFIG_DIR'] = str(directory/'ultralytics')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    from fastapi.testclient import TestClient
    from main import app
    from backend.context import DatasetContext, Goal
    from backend.workbench.api import services
    from sql.store import Store
    from backend.workbench.datasets import image_source, load
    services.cache_clear()
    print('Loaded application; opening isolated experiment store', flush=True)

    def post(client, path, body):
        response = client.post('/api/v1'+path, json=body)
        response.raise_for_status()
        return response.json()

    def wait(client, job):
        last = None
        while time.monotonic()-started < args.timeout:
            response = client.get('/api/v1/jobs/'+job['id'])
            response.raise_for_status()
            record = response.json()
            payload = record['payload']
            if payload['stage'] != last:
                last = payload['stage']
                print(f"{payload['task']}: {last}", flush=True)
            if payload['status'] == 'completed':
                return record
            if payload['status'] not in ('queued', 'running'):
                raise RuntimeError(str(payload)+'\n'+client.get('/api/v1/jobs/'+job['id']+'/logs').text)
            time.sleep(.5)
        client.post('/api/v1/jobs/'+job['id']+'/cancel')
        raise TimeoutError('Experiment budget exceeded; active job cancelled')

    def upload(client, path):
        with path.open('rb') as incoming:
            response = client.post('/api/v1/datasets/import', files={'file': (path.name, incoming, 'application/zip')})
        response.raise_for_status()
        return response.json()

    with TestClient(app) as client:
        print('API ready; importing baseline and mock data', flush=True)
        store, _ = services()
        origin = Store(args.baseline_store).get(args.baseline_model, 'model')
        baseline = store.create('model', {**copy.deepcopy(origin['payload']),
            'experiment_role': 'Unchanged local synthetic checkpoint; not general-purpose pretrained weights',
            'source_model_id': origin['id'], 'source_store': str(Path(args.baseline_store).resolve())})
        clean = upload(client, fixture(directory/'corrected.zip', seed=20261101))
        clean_split = post(client, '/datasets/'+clean['id']+'/splits', {'group_column': 'flight_id', 'seed': 42})
        raw = post(client, '/variations', {'dataset_id': clean['id'], 'split_id': clean_split['id'],
                   'seed': 42, 'intervention': 'label_omission', 'severity': .5})
        raw_split = post(client, '/datasets/'+raw['id']+'/splits', {'group_column': 'flight_id', 'seed': 42})
        # Audit paired image bytes and untouched held-out annotations across the intervention.
        raw_data, clean_data = load(store, raw['id']), load(store, clean['id'])
        assert sorted(raw_data.image_hashes.values()) == sorted(clean_data.image_hashes.values())
        def held_out(dataset, split):
            result = []
            with image_source(dataset['payload']) as archive:
                for key in ('val', 'test'):
                    for name in split['payload']['images'][key]:
                        label = name.replace('images/', 'labels/', 1).rsplit('.', 1)[0]+'.txt'
                        # Canonical variants may reformat floats; compare parsed numeric labels.
                        rows = sorted(tuple(float(v) for v in line.split()) for line in archive.read(label).decode().splitlines() if line.strip())
                        result.append((key, hashlib.sha256(archive.read(name)).hexdigest(), rows))
            return sorted(result)
        assert held_out(raw, raw_split) == held_out(clean, clean_split)
        model_ids = {'baseline': baseline['id']}
        readiness = {}
        context = DatasetContext(original_text='Detect drone-shaped synthetic rectangles. Software experiment only.',
            goal=Goal(task_type='object_detection', required_classes=['drone'], input_modalities=['image']), confirmed=True)
        for role, dataset, split in [('raw_labels', raw, raw_split), ('corrected_labels', clean, clean_split)]:
            ready = wait(client, post(client, '/readiness', {'dataset_id': dataset['id'], 'split_id': split['id'],
                                                        'context': context.model_dump()}))
            readiness[role] = {'id': ready['id'], 'status': ready['payload']['result']['overall_status']}
            trained = wait(client, post(client, '/training', {'dataset_id': dataset['id'], 'split_id': split['id'],
                'readiness_id': ready['id'], 'model_type': 'controlled' if role == 'raw_labels' else 'candidate',
                'initialization': 'pretrained', 'weights_path': baseline['payload']['artifact'],
                'epochs': args.epochs, 'batch': 8, 'image_size': 64, 'learning_rate': .02, 'seed': 42,
                'device': 'cpu', 'experiment_group': 'quick-synthetic-label-quality'}))
            model_ids[role] = trained['payload']['result']['model_id']
        a, b = [store.get(model_ids[role])['payload'] for role in ('raw_labels', 'corrected_labels')]
        assert a['configuration']['initialization_sha256'] == b['configuration']['initialization_sha256']
        # Generate final testing data only after both runs finish. Never tune using it.
        test = upload(client, fixture(directory/'external-test.zip', seed=20261102, external=True))
        evaluated = wait(client, post(client, '/evaluations', {'model_ids': list(model_ids.values()),
            'dataset_id': test['id'], 'device': 'cpu', 'confidence': .25, 'match_iou': .5, 'iou': .7}))
        evaluation_ids = evaluated['payload']['result']['evaluation_ids']
        evaluations = [store.get(id, 'evaluation')['payload'] for id in evaluation_ids]
        assert len({e['test_sha256'] for e in evaluations}) == 1
        assert all(e['test_images'] == 20 for e in evaluations)
        by_role = {role: evaluation['metrics'] for role, evaluation in zip(model_ids, evaluations)}
        deltas = {reference: {metric: None if by_role['corrected_labels'].get(metric) is None or by_role[reference].get(metric) is None
                  else 100*(by_role['corrected_labels'][metric]-by_role[reference][metric])
                  for metric in ('mAP50', 'mAP50_95', 'precision', 'recall', 'f1')}
                  for reference in ('baseline', 'raw_labels')}
        summary = {'status': 'passed', 'elapsed_seconds': round(time.monotonic()-started, 2),
            'scope': 'Synthetic drawings only; one seed; generator-truth correction, not automatic repair or real UAV validation.',
            'baseline': {'source_store': args.baseline_store, 'source_model_id': args.baseline_model,
                         'description': 'Existing trusted local workbench checkpoint, unchanged',
                         'original_training_epochs': origin['payload']['configuration'].get('epochs')},
            'training': {'epochs_each': args.epochs, 'train_images': 32, 'validation_images': 8, 'reserved_internal_test_images': 8,
                         'external_test_images': 20, 'image_size': 64, 'batch': 8, 'learning_rate': .02, 'seed': 42,
                         'raw_train_objects': len([o for o in raw_data.objects if o['image'] in raw_split['payload']['images']['train']]),
                         'corrected_train_objects': 32},
            'readiness': readiness, 'model_ids': model_ids, 'evaluation_ids': evaluation_ids,
            'test_sha256': evaluations[0]['test_sha256'], 'metrics': by_role,
            'corrected_minus_reference_percentage_points': deltas,
            'audits': {'same_finetuning_initialization': True, 'same_input_images': True,
                       'same_validation_and_internal_test_labels': True, 'same_external_test_and_thresholds': True,
                       'development_test_leakage_checks': 'passed by evaluation adapter'},
            'limitations': ['Readiness checks cannot discover deliberately omitted objects without trusted references.',
                            'Improvement, if observed, belongs to this controlled correction/fine-tuning experiment.',
                            'No general-purpose pretrained checkpoint was available; baseline is a weak synthetic model.',
                            'The unchanged baseline uses its original inference checkpoint; fine-tuning follows adapter initialization.']}
        (directory/'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        with (directory/'results.csv').open('w', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=['model', 'mAP50', 'mAP50_95', 'precision', 'recall', 'f1', 'true_positive', 'false_positive', 'false_negative'])
            writer.writeheader()
            for role, metrics in by_role.items():
                writer.writerow({'model': role, **{k: metrics.get(k) for k in writer.fieldnames if k != 'model'}})
        rows = ['# Quick synthetic accuracy experiment', '',
                'One seed; mock drawings; corrected labels come from known generator truth. No automatic repair is claimed.', '',
                '| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 | TP / FP / FN |',
                '|---|---:|---:|---:|---:|---:|---|']
        def pct(value):
            return 'undefined' if value is None else f'{100*value:.2f}%'
        for role, metrics in by_role.items():
            rows.append('| '+role+' | '+' | '.join(pct(metrics.get(k)) for k in ('mAP50','mAP50_95','precision','recall','f1'))+
                        f" | {metrics['true_positive']} / {metrics['false_positive']} / {metrics['false_negative']} |")
        rows += ['', f"32 training images, 8 validation images, 20 independent final test images; {args.epochs} epochs per fine-tuning arm.",
                 f"Elapsed: {summary['elapsed_seconds']} seconds. CPU only.", '',
                 'mAP uses a confidence sweep with floor 0.001. Precision/recall/F1 use confidence 0.25 and matching IoU 0.5.', '',
                 'Corrected-label minus baseline mAP@50: '+str(round(deltas['baseline']['mAP50'], 4))+' percentage points.',
                 'Corrected-label minus raw-label mAP@50: '+str(round(deltas['raw_labels']['mAP50'], 4))+' percentage points.', '',
                 'Readiness outcomes: '+json.dumps(readiness)+'. These checks cannot establish that every visible object was annotated.', '',
                 '## Limitations', '']
        rows.extend('- '+line for line in summary['limitations'])
        rows += ['', 'Machine-readable evidence: results.json and results.csv. Full API/worker evidence: storage/.', '']
        (directory/'report.md').write_text('\n'.join(rows), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
        print('Report: '+str(directory/'report.md'), flush=True)


if __name__ == '__main__':
    main()
