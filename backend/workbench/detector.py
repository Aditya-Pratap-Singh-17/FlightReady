"""Local Ultralytics YOLO11 adapter. No dataset or weight URL resolution."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import time
import numpy as np
import yaml
from backend.workbench.datasets import load, image_source, file_sha
from backend.workbench.splits import get_split, signatures, assert_independent, assert_development_disjoint


def compute():
    import importlib.util
    result = {'torch_installed': bool(importlib.util.find_spec('torch')), 'detector_installed': bool(importlib.util.find_spec('ultralytics')),
              'cuda_available': False, 'devices': ['cpu'], 'selected_device': 'cpu'}
    if result['torch_installed']:
        try:
            import torch
            result.update(torch_version=torch.__version__, cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available())
            if torch.cuda.is_available():
                result.update(devices=['cpu', '0'], selected_device='0', gpu_name=torch.cuda.get_device_name(0),
                              gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        except (ImportError, OSError, RuntimeError) as exc:
            result['error'] = str(exc)
    return result


def device(value):
    available = compute()
    if not available['torch_installed'] or not available['detector_installed']:
        raise ValueError('Training dependencies are missing. Install requirements/requirements-training.txt in the backend environment.')
    selected = available['selected_device'] if value == 'auto' else value
    if selected == '0' and not available['cuda_available']:
        raise ValueError('CUDA is unavailable in this PyTorch runtime. Select CPU or install a CUDA-enabled PyTorch build.')
    return selected


def gpu_usage(torch, selected):
    if selected == 'cpu' or not torch.cuda.is_available():
        return None
    return {'allocated_bytes': torch.cuda.memory_allocated(0), 'reserved_bytes': torch.cuda.memory_reserved(0),
            'total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'scope': 'PyTorch worker memory at the last completed epoch, not GPU utilization'}


def resume_training(store, job_id, config):
    """Resume only an existing epoch checkpoint, retaining optimizer/EMA state."""
    source_id = config['resume_from_job_id']
    source = store.get(source_id, 'job')['payload']
    validate_training(store, config)
    directory = Path(source.get('run_directory', store.directory('jobs', source_id) / 'run')).parent
    checkpoint = directory / 'run' / 'weights' / 'last.pt'
    frozen = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
    for key in ('dataset_id', 'split_id', 'readiness_id', 'architecture', 'epochs', 'seed', 'batch', 'image_size', 'learning_rate'):
        if config.get(key) != source['config'].get(key):
            raise ValueError('Resume must retain the frozen training protocol')
    if not checkpoint.is_file():
        raise ValueError('No resumable epoch checkpoint exists')
    selected = device(config['device'])
    model = setup_detector()(str(checkpoint), task='detect')
    import torch
    def update(trainer):
        if int(trainer.epoch)+1 > config['epochs']:
            return
        store.update(job_id, stage=f'Resumed epoch {trainer.epoch+1}/{config["epochs"]}',
            progress={'epoch': int(trainer.epoch)+1, 'epochs': config['epochs'], 'gpu': gpu_usage(torch, selected)})
    model.add_callback('on_fit_epoch_end', update)
    store.update(job_id, stage='Restoring checkpoint and optimizer state', run_directory=str(directory / 'run'))
    model.train(resume=True, device=selected, workers=0)
    manifests = [get_split(store, config['split_id'], config['dataset_id'])]
    if config['model_type'] == 'combined':
        manifests.append(get_split(store, config['reference_split_id'], config['reference_dataset_id']))
    validation=validation_source(store,config,frozen['classes'])
    if validation:
        manifests=[training_only_manifest(m) for m in manifests]+[validation['manifest']]
    counts = {m['dataset_id']: len(m['images']['train']) for m in manifests if m['images']['train']}
    return finish_training(store, job_id, config, directory, {**frozen, 'resumed_from_job_id': source_id},
                           frozen['classes'], manifests, counts, directory / 'data.yaml', selected)


def classes_for(store, id):
    payload = store.get(id, 'dataset')['payload']
    if not payload['training']['eligible']:
        raise ValueError('; '.join(payload['training']['reasons']))
    return list(payload['profile'].get('declared_classes') or payload['profile']['class_counts'])


def class_mapping(source, target, mapping):
    if not mapping and set(source) == set(target):
        return {name: name for name in source}
    if set(mapping) != set(source) or set(mapping.values()) != set(target) or len(set(mapping.values())) != len(mapping):
        raise ValueError('Class schemas differ. Supply an explicit complete one-to-one source-to-model class mapping; classes are never silently dropped.')
    return mapping


def materialize(store, dataset_id, names, target_classes, mapping):
    """Cache one decoded-source copy; reuse image bytes with filesystem hard links."""
    ds = load(store, dataset_id)
    payload = store.get(dataset_id, 'dataset')['payload']
    source_classes = classes_for(store, dataset_id)
    mapping = class_mapping(source_classes, target_classes, mapping)
    key = hashlib.sha256(json.dumps([dataset_id, target_classes, mapping], sort_keys=True).encode()).hexdigest()
    root = store.directory('cache', key)
    (root / 'images').mkdir(exist_ok=True)
    (root / 'labels').mkdir(exist_ok=True)
    objects = {}
    for obj in ds.objects:
        objects.setdefault(obj['image'], []).append(obj)
    output = []
    with image_source(payload) as archive:
        for name in sorted(names):
            raw = archive.read(name)
            expected = ds.image_hashes[name]
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Dataset changed since import; create a new snapshot')
            stem = hashlib.sha256(name.encode()).hexdigest()
            path = root / 'images' / (stem + Path(name).suffix.lower())
            if not path.exists():
                if payload['source_kind'] == 'directory':
                    # External files are mutable: cache verified bytes rather than linking them.
                    path.write_bytes(raw)
                else:
                    shared = store.directory('imagecache', dataset_id) / (expected + Path(name).suffix.lower())
                    if not shared.exists():
                        shared.write_bytes(raw)
                    try:
                        os.link(shared, path)
                    except OSError:
                        path.write_bytes(raw)
            elif file_sha(path) != expected:
                raise ValueError('Managed image cache checksum mismatch')
            labels = [' '.join([str(target_classes.index(mapping[o['class']]))] + [format(float(v), '.12g') for v in o['bbox']]) for o in objects.get(name, [])]
            (root / 'labels' / (stem + '.txt')).write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
            output.append((name, path))
    return output


def write_yaml(directory, classes, splits):
    data = {'path': str(directory), 'names': classes}
    for split, paths in splits.items():
        listing = directory / (split + '.txt')
        listing.write_text('\n'.join(str(path) for path in paths) + '\n', encoding='utf-8')
        data[split] = str(listing)
    path = directory / 'data.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def setup_detector():
    os.environ['YOLO_OFFLINE'] = 'true'
    config_directory=Path(os.getenv('WORKBENCH_ROOT','.runtime/workbench')).resolve()/'ultralytics'
    config_directory.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(config_directory)
    from ultralytics import YOLO, settings
    settings.update({'sync': False, 'wandb': False, 'comet': False, 'mlflow': False, 'clearml': False, 'tensorboard': False})
    return YOLO


def validation_source(store, config, classes):
    """An optional separate validation source; never substitute the final test."""
    id=config.get('validation_dataset_id')
    if not id:
        if config.get('validation_split_id'):
            raise ValueError('A validation split requires a validation dataset')
        return None
    mapping=class_mapping(classes_for(store,id),classes,config.get('validation_class_mapping',{}))
    if config.get('validation_split_id'):
        manifest=get_split(store,config['validation_split_id'],id)
        names=manifest['images']['val']
    else:
        manifest,names=test_manifest(store,{'dataset_id':id})
    ds=load(store,id)
    if not any(o['image'] in names for o in ds.objects):
        raise ValueError('Selected validation data has no annotated objects')
    from metrics.quality_profiles import fingerprint
    manifest={**manifest,'dataset_id':id,'dataset_sha256':ds.sha256,
              'images':{'train':[],'val':sorted(names),'test':[]}}
    manifest['checksum']=fingerprint({'dataset_sha256':ds.sha256,'validation_images':sorted((n,ds.image_hashes[n]) for n in names)})
    return {'manifest':manifest,'names':names,'mapping':mapping}


def training_only_manifest(manifest):
    return {**manifest,'images':{**manifest['images'],'val':[]}}


def validate_training(store, config):
    classes = classes_for(store, config['dataset_id'])
    if config['model_type']!='combined' and config.get('class_mapping'):
        target=[config['class_mapping'].get(name) for name in classes]
        if any(not isinstance(name,str) or not name.strip() for name in target):
            raise ValueError('Class mappings need complete nonempty canonical target names')
        class_mapping(classes,target,config['class_mapping'])
        classes=target
    split = get_split(store, config['split_id'], config['dataset_id'])
    if not split['images']['train']:
        raise ValueError('A nonempty labeled training partition is required')
    if not split['images']['val'] and not config.get('validation_dataset_id'):
        raise ValueError('Training-only manifests require a separate labeled validation dataset for checkpoint selection; the final test cannot be used for selection')
    readiness = store.get(config['readiness_id'], 'job')['payload']
    if readiness['task'] != 'readiness' or readiness['status'] != 'completed':
        raise ValueError('A completed training-subset readiness report is required')
    if readiness['config']['dataset_id'] != config['dataset_id'] or readiness['config'].get('split_id') != config['split_id']:
        raise ValueError('Readiness report must match this exact dataset and training split')
    if readiness.get('result', {}).get('confirmed_context', {}).get('goal', {}).get('task_type', 'object_detection') != 'object_detection':
        raise ValueError('Detector training requires an object-detection readiness context')
    if config.get('aggregation_id'):
        aggregation=store.get(config['aggregation_id'],'aggregation')['payload']
        if aggregation['request']['readiness_id']!=config['readiness_id'] or aggregation['result']['task']!='object_detection' or aggregation['result']['requires_measurement']:
            raise ValueError('Selected aggregation must match the exact detection readiness assessment')
    ds=load(store,config['dataset_id'])
    from backend.compatibility import compatibility
    from backend.context import DatasetContext
    context = readiness.get('result', {}).get('confirmed_context')
    if context:
        preview = compatibility(ds, DatasetContext.model_validate(context))
        if not preview['training_partition_available']:
            raise ValueError('A validation-only/test-only source cannot be used for training; supply separate labeled training data.')
        if preview['status'] != 'supported':
            raise ValueError(preview['summary'] + ' Resolve required labels and operating-condition evidence before training for this goal.')
    if not any(o['image'] in split['images']['train'] for o in ds.objects):
        raise ValueError('Training partition has no annotated objects')
    if not config.get('validation_dataset_id') and not any(o['image'] in split['images']['val'] for o in ds.objects):
        raise ValueError('Validation partition has no annotated objects; mAP-based checkpoint selection would be undefined')
    if config['model_type'] == 'controlled' and not store.get(config['dataset_id'])['payload'].get('provenance', {}).get('artificial'):
        raise ValueError('Controlled models require a recorded controlled dataset variation')
    if config['model_type'] == 'combined':
        if not config.get('reference_dataset_id') or not config.get('reference_split_id'):
            raise ValueError('Combined training requires a reference dataset and split')
        reference = get_split(store, config['reference_split_id'], config['reference_dataset_id'])
        class_mapping(classes_for(store, config['reference_dataset_id']), classes, config['class_mapping'])
        # No source's held-out data may appear in the other source's development data.
        assert_independent([split], reference, reference['images']['test'], acknowledge=True)
        assert_independent([reference], split, split['images']['test'], acknowledge=True)
        assert_development_disjoint(split,reference)
        train_hashes = {split['hashes'][n] for n in split['images']['train']}
        ref_val_hashes = {reference['hashes'][n] for n in reference['images']['val']}
        ref_train_hashes = {reference['hashes'][n] for n in reference['images']['train']}
        val_hashes = {split['hashes'][n] for n in split['images']['val']}
        if train_hashes & ref_val_hashes or ref_train_hashes & val_hashes:
            raise ValueError('Combined sources leak training images into validation')
    if config['initialization'] == 'pretrained':
        weights = Path(config.get('weights_path') or '').resolve()
        if not weights.is_file() or weights.suffix.lower() != '.pt':
            raise ValueError('Choose a trusted existing local .pt checkpoint; weight downloads are explicit outside training')
    validation=validation_source(store,config,classes)
    if validation:
        development=[training_only_manifest(split)]
        if config['model_type']=='combined':
            development.append(training_only_manifest(reference))
        assert_independent(development,validation['manifest'],validation['names'],config.get('acknowledge_validation_groups',False))
    return classes, split


def train(store, job_id, config):
    if config.get('resume_from_job_id'):
        return resume_training(store, job_id, config)
    classes, split = validate_training(store, config)
    selected_device = device(config['device'])
    directory = store.directory('jobs', job_id)
    store.update(job_id, stage='Preparing immutable training and validation manifests')
    candidate_mapping={} if config['model_type']=='combined' else config.get('class_mapping',{})
    validation_source_data=validation_source(store,config,classes)
    splits = {key: [path for _, path in materialize(store, config['dataset_id'], split['images'][key], classes, candidate_mapping)] for key in (('train',) if validation_source_data else ('train', 'val'))}
    manifests = [training_only_manifest(split) if validation_source_data else split]
    counts = {config['dataset_id']: len(split['images']['train'])}
    if config['model_type'] == 'combined':
        reference = get_split(store, config['reference_split_id'], config['reference_dataset_id'])
        manifests.append(training_only_manifest(reference) if validation_source_data else reference)
        counts[config['reference_dataset_id']] = len(reference['images']['train'])
        for key in splits:
            splits[key].extend(path for _, path in materialize(store, config['reference_dataset_id'], reference['images'][key], classes, config['class_mapping']))
    if validation_source_data:
        splits['val']=[path for _,path in materialize(store,config['validation_dataset_id'],validation_source_data['names'],classes,validation_source_data['mapping'])]
        manifests.append(validation_source_data['manifest'])
    if config['model_type']=='combined':
        from backend.workbench.combined import combined_readiness
        store.update(job_id,stage='Assessing the complete combined training evidence')
        config={**config,**combined_readiness(store,config,classes)}
        store.update(job_id,config=config)
    data = write_yaml(directory, classes, splits)
    import torch
    YOLO = setup_detector()
    torch.manual_seed(0)  # matched scratch initialization across repeated training seeds
    if config['initialization'] == 'pretrained':
        checkpoint = Path(config['weights_path']).resolve()
        checkpoint_hash = file_sha(checkpoint)
        model = YOLO(str(checkpoint), task='detect')
        declared=str(model.model.yaml.get('yaml_file',''))
        if 'yolo11' not in declared or model.model.yaml.get('scale', 'n') != config['architecture'][-1]:
            raise ValueError('Checkpoint architecture does not match selected architecture')
    else:
        checkpoint_hash = None
        model = YOLO(config['architecture'] + '.yaml', task='detect')
    # Initialize the target class head under a fixed seed before the trainer sets
    # the run's augmentation/data-order seed. Transfer the COMPLETE target model.
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(0)
    initialized=DetectionModel(model.model.yaml,nc=len(classes),ch=3,verbose=False)
    if config['initialization']=='pretrained':
        initialized.load(model.model,verbose=False)
    initialized.names=dict(enumerate(classes))
    model.model=initialized
    # A YAML wrapper otherwise rebuilds using the run seed. Reload a local
    # checkpoint to give both modes an explicit, identical initialization.
    initialization_path=directory/'initialization.pt'
    model.save(str(initialization_path))
    model=YOLO(str(initialization_path),task='detect')
    initialized=model.model
    def tensor_hash(network):
        digest=hashlib.sha256()
        for name,value in network.state_dict().items():
            digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    initial_hash=tensor_hash(initialized)
    initial_state={k:v.detach().cpu().clone() for k,v in initialized.state_dict().items()}
    # Persist frozen configuration before compute starts; hashes catch local weight changes.
    frozen = {**config, 'initialization_sha256': initial_hash, 'checkpoint_sha256': checkpoint_hash, 'selected_device': selected_device,
              'classes': classes, 'training_sizes': counts, 'split_checksums': [s['checksum'] for s in manifests]}
    (directory / 'configuration.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    def epoch_end(trainer):
        epoch = int(trainer.epoch) + 1
        if epoch > config['epochs']:
            return  # Ultralytics invokes fit-end again after final best-weight validation.
        losses = trainer.label_loss_items(trainer.tloss, prefix='train')
        store.update(job_id, stage=f'Epoch {epoch}/{config["epochs"]}', progress={'epoch': epoch, 'epochs': config['epochs'], 'losses': {k: float(v) for k, v in losses.items()}, 'validation': {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (float, int))}, 'gpu': gpu_usage(torch, selected_device)})
        print(json.dumps({'epoch': epoch, 'losses': losses}), flush=True)
    model.add_callback('on_fit_epoch_end', epoch_end)
    def verify_initialization(trainer):
        actual=tensor_hash(trainer.model)
        if actual!=initial_hash:
            actual_state=trainer.model.state_dict()
            differences=[k for k,v in initial_state.items() if k not in actual_state or not torch.equal(v,actual_state[k].detach().cpu())]
            differences.extend(k for k in actual_state if k not in initial_state)
            raise ValueError('Trainer initialization differs from recorded weights: '+str(differences))
        print('Verified initial target-model SHA256: '+actual,flush=True)
    model.add_callback('on_train_start',verify_initialization)
    store.update(job_id, stage='Training detector')
    model.train(data=str(data), epochs=config['epochs'], batch=config['batch'], imgsz=config['image_size'],
                lr0=config['learning_rate'], seed=config['seed'], device=selected_device, workers=0,
                deterministic=True, optimizer='SGD', pretrained=True, amp=False, cache=False, patience=0,
                project=str(directory), name='run', exist_ok=False, plots=False, save=True, verbose=True)
    return finish_training(store, job_id, config, directory, frozen, classes, manifests, counts, data, selected_device)


def finish_training(store, job_id, config, directory, frozen, classes, manifests, counts, data, selected_device):
    artifact = directory / 'run' / 'weights' / 'best.pt'
    if not artifact.is_file():
        raise ValueError('Training completed without a best checkpoint')
    from backend.research.experiments import _metrics
    YOLO = setup_detector()
    validation = _metrics(YOLO(str(artifact), task='detect').val(data=str(data), split='val', device=selected_device,
        imgsz=config['image_size'], batch=config['batch'], workers=0, max_det=300, conf=.001, iou=.7,
        plots=False, project=str(directory), name='validation'))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        revision = None
    # Hash actual source as well: worktrees may be dirty or have no commit yet.
    code = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    source_paths = list(root.glob('*.py')) + [path for folder in ('backend', 'metrics', 'sql')
        for path in (root/folder).rglob('*.py') if '__pycache__' not in path.parts]
    for path in sorted(source_paths):
        code.update(path.relative_to(root).as_posix().encode())
        code.update(path.read_bytes())
    effective_readiness_id=config.get('combined_readiness_id') or config['readiness_id']
    payload = {'job_id': job_id, 'dataset_id': config.get('combined_dataset_id') or config['dataset_id'], 'readiness_id': effective_readiness_id, 'configuration': frozen,
               'artifact': str(artifact), 'artifact_sha256': file_sha(artifact), 'validation': validation,
               'validation_settings':{'nms_iou':.7,'ap_confidence_floor':.001,'image_size':config['image_size'],'max_detections':300,'device':selected_device},
               'development_manifests': manifests, 'code_version': revision, 'code_sha256': code.hexdigest(),
               'environment': {'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p) for p in ('torch', 'ultralytics', 'numpy')}},
               'training_images': sum(counts.values()),
               'unique_training_image_hashes':len({manifest['hashes'][name] for manifest in manifests for name in manifest['images']['train']}),
               'sampling_proportions': {k: v/sum(counts.values()) for k, v in counts.items()}}
    payload['validation_dataset_id']=config.get('validation_dataset_id') or config['dataset_id']
    payload['quality_profile'] = (store.get(effective_readiness_id,'job')['payload']['result'].get('quality_profile') if config.get('combined_readiness_id') else
                                  store.get(config['aggregation_id'],'aggregation')['payload']['result'] if config.get('aggregation_id')
                                  else store.get(config['readiness_id'],'job')['payload']['result'].get('quality_profile'))
    payload['quality_evidence_scope'] = ('Complete combined training manifests; related combined runs remain excluded from independent-dataset correlations.'
        if config['model_type']=='combined' else 'Exact selected training subset')
    record = store.create('model', payload)
    return {'model_id': record['id'], 'validation': validation, 'artifact_sha256': payload['artifact_sha256']}


def iou(a, b):
    def xy(box):
        x, y, w, h = box
        return x-w/2, y-h/2, x+w/2, y+h/2
    ax, ay, ar, ab = xy(a)
    bx, by, br, bb = xy(b)
    intersection = max(0, min(ar, br)-max(ax, bx)) * max(0, min(ab, bb)-max(ay, by))
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return intersection/union if union > 0 else 0


def detection_errors(truth, predictions, threshold):
    used = set()
    annotations = []
    for prediction in sorted(predictions, key=lambda p: -p['confidence']):
        candidates = [(iou(prediction['bbox'], t['bbox']), i) for i, t in enumerate(truth) if i not in used and t['class'] == prediction['class']]
        match = max(candidates, default=(0, -1))
        matched = match[0] >= threshold
        if matched:
            used.add(match[1])
        annotations.append({**prediction, 'outcome': 'true_positive' if matched else 'false_positive'})
    missed = [{**t, 'outcome': 'false_negative'} for i, t in enumerate(truth) if i not in used]
    tp = len(used)
    fp = len(predictions)-tp
    fn = len(truth)-tp
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'precision': precision, 'recall': recall, 'f1': f1}, annotations, missed


def native_size_outcomes(truth, predictions, missed, width, height):
    """Operating-point recall by ground-truth area, not small-object AP.

    Unmatched predictions have no true object size; their size is kept separate.
    """
    from metrics.uav_metrics.research import DEFAULT_BINS, bin_name
    size=lambda item: bin_name(item['bbox'][2]*item['bbox'][3]*width*height,DEFAULT_BINS)
    population=Counter(size(item) for item in truth)
    absent=Counter(size(item) for item in missed)
    false=Counter(size(item) for item in predictions if item['outcome']=='false_positive')
    return {name:{'ground_truth_objects':population[name],'false_negative':absent[name],
                  'true_positive':population[name]-absent[name], 'false_positives_by_predicted_area':false[name],
                  'recall':(population[name]-absent[name])/population[name] if population[name] else None} for name in DEFAULT_BINS}


def confusion_counts(truth, predictions, classes, threshold):
    """Confidence-ordered spatial matching, ignoring class to expose confusion."""
    matrix=np.zeros((len(classes)+1,len(classes)+1),dtype=int)
    used=set()
    for prediction in sorted(predictions,key=lambda p:-p['confidence']):
        candidates=[(iou(prediction['bbox'],t['bbox']),i) for i,t in enumerate(truth) if i not in used]
        match=max(candidates,default=(0,-1))
        actual=len(classes)
        if match[0]>=threshold:
            used.add(match[1]);actual=classes.index(truth[match[1]]['class'])
        matrix[classes.index(prediction['class']),actual]+=1
    for i,t in enumerate(truth):
        if i not in used:
            matrix[len(classes),classes.index(t['class'])]+=1
    return matrix


def test_manifest(store, config):
    ds = load(store, config['dataset_id'])
    if config.get('split_id'):
        manifest = get_split(store, config['split_id'], config['dataset_id'])
        names = manifest['images']['test']
    else:
        group_column = next((c for c in ('flight_id', 'session_id', 'recording_id', 'video_id') if c in ds.frame and not ds.frame[c].isna().any()), None)
        groups = {r['image']: str(r[group_column]) for r in ds.frame.to_dict('records')} if group_column else {}
        manifest = {'hashes': ds.image_hashes, 'perceptual': signatures(store, config['dataset_id']), 'groups': groups,
                    'group_column': group_column, 'config': {'near_distance': 4}}
        names = sorted(ds.image_hashes)
    manifest = {**manifest, 'dataset_sha256': ds.sha256}
    images=sorted((n,ds.image_hashes[n]) for n in names)
    manifest['image_checksum']=hashlib.sha256(json.dumps(images).encode()).hexdigest()
    selected=set(names)
    mapping=config.get('class_mapping',{})
    annotations=sorted((o['image'],mapping.get(o['class'],o['class']),tuple(o['bbox'])) for o in ds.objects if o['image'] in selected)
    schema=sorted(mapping.get(c,c) for c in (ds.profile.get('declared_classes') or ds.profile.get('class_counts',{})))
    manifest['test_checksum'] = hashlib.sha256(json.dumps({'images':images,'annotations':annotations,
        'classes':schema,'ground_truth_protocol':'canonical_boxes_1.0'},sort_keys=True).encode()).hexdigest()
    return manifest, names


def evaluate_models(store, job_id, config):
    selected_device = device(config['device'])
    source_classes = classes_for(store, config['dataset_id'])
    manifest, names = test_manifest(store, config)
    models = [store.get(id, 'model') for id in config['model_ids']]
    for record in models:
        assert_independent(record['payload']['development_manifests'], manifest, names, config['acknowledge_missing_groups'])
        class_mapping(source_classes, record['payload']['configuration']['classes'], config['class_mapping'])
    directory = store.directory('jobs', job_id)
    YOLO = setup_detector()
    from backend.research.experiments import _metrics
    ds = load(store, config['dataset_id'])
    results = []
    for record in models:
        model_id, payload = record['id'], record['payload']
        store.update(job_id, stage='Evaluating ' + model_id[:8])
        if file_sha(payload['artifact']) != payload['artifact_sha256']:
            raise ValueError('Model artifact checksum mismatch')
        classes = payload['configuration']['classes']
        mapping = class_mapping(source_classes, classes, config['class_mapping'])
        images = materialize(store, config['dataset_id'], names, classes, mapping)
        model_dir = directory / model_id
        model_dir.mkdir()
        paths = [path for _, path in images]
        data = write_yaml(model_dir, classes, {'train': paths, 'val': paths, 'test': paths})
        model = YOLO(payload['artifact'], task='detect')
        measured = model.val(data=str(data), split='test', device=selected_device, conf=.001, iou=config['iou'],
            imgsz=payload['configuration']['image_size'], batch=1, workers=0, max_det=300, plots=False, project=str(model_dir), name='metrics')
        metrics = _metrics(measured)
        curves = []
        observed_class_ids = [int(i) for i in getattr(measured.box, 'ap_class_index', [])]
        for x, y, xlabel, ylabel in getattr(measured, 'curves_results', []):
            x, y = np.asarray(x), np.asarray(y)
            if x.size and y.size:
                # All points are genuine validator output, uniformly downsampled
                # for transport; no curve is inferred from an aggregate metric.
                indexes = np.unique(np.linspace(0, len(x)-1, min(101, len(x)), dtype=int))
                curves.append({'x_label': xlabel, 'y_label': ylabel, 'x': x[indexes].tolist(),
                    'series': [{'class': classes[observed_class_ids[i]], 'values': values[indexes].tolist()}
                               for i, values in enumerate(np.atleast_2d(y)) if i < len(observed_class_ids)]})
        metrics['per_class_mAP50_95']={classes[int(k)]:v for k,v in metrics['per_class_mAP50_95'].items()}
        if not any(o['image'] in names for o in ds.objects):
            metrics.update(mAP50=None,mAP50_95=None,per_class_mAP50_95={})
        totals = Counter()
        per_class_counts = {name: Counter() for name in classes}
        confusion=np.zeros((len(classes)+1,len(classes)+1),dtype=int)
        predictions = []
        latency = []
        size_totals=defaultdict(Counter)
        dimensions=ds.frame.set_index('image')[['width','height']].to_dict('index')
        for index, (name, path) in enumerate(images):
            result = model.predict(str(path), device=selected_device, conf=config['confidence'], iou=config['iou'],
                                   imgsz=payload['configuration']['image_size'], max_det=300, verbose=False)[0]
            detected = [{'class': classes[int(box.cls.item())], 'confidence': float(box.conf.item()),
                         'bbox': [float(v) for v in box.xywhn[0].tolist()]} for box in result.boxes]
            truth = [{**o, 'class': mapping[o['class']]} for o in ds.objects if o['image'] == name]
            errors, annotated, missed = detection_errors(truth, detected, config['match_iou'])
            size_evidence=native_size_outcomes(truth,annotated,missed,dimensions[name]['width'],dimensions[name]['height'])
            for size,values in size_evidence.items():
                size_totals[size].update({k:v for k,v in values.items() if k!='recall'})
            confusion+=confusion_counts(truth,detected,classes,config['match_iou'])
            totals.update({k: errors[k] for k in ('true_positive', 'false_positive', 'false_negative')})
            for item in annotated + missed:
                per_class_counts[item['class']][item['outcome']] += 1
            latency.append(float(result.speed['inference']))
            predictions.append({'image': name, 'ground_truth': truth, 'predictions': annotated, 'missed': missed})
            store.update(job_id, stage=f'Inference {index+1}/{len(images)} for {model_id[:8]}', progress={'images': index+1, 'total_images': len(images)})
        tp, fp, fn = (totals[k] for k in ('true_positive', 'false_positive', 'false_negative'))
        metrics.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                       f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None, **dict(totals),
                       inference_latency_ms=float(np.mean(latency)), latency_sample_size=len(latency))
        metrics.update(false_positives_per_image=fp/len(images), false_negative_fraction=fn/(tp+fn) if tp+fn else None,
                       ground_truth_objects=tp+fn, predicted_objects=tp+fp)
        for size,values in size_totals.items():
            metrics['recall_'+size+'_native']=values['true_positive']/values['ground_truth_objects'] if values['ground_truth_objects'] else None
        per_class = {}
        for name, values in per_class_counts.items():
            a, b, c = (values[k] for k in ('true_positive', 'false_positive', 'false_negative'))
            per_class[name] = {'true_positive': a, 'false_positive': b, 'false_negative': c,
                'precision': a/(a+b) if a+b else None, 'recall': a/(a+c) if a+c else None,
                'f1': 2*a/(2*a+b+c) if 2*a+b+c else None,
                'mAP50_95': metrics['per_class_mAP50_95'].get(name)}
        prediction_path = model_dir / 'predictions.json'
        prediction_path.write_text(json.dumps(predictions, allow_nan=False), encoding='utf-8')
        evaluation = store.create('evaluation', {'job_id': job_id, 'model_id': model_id, 'dataset_id': config['dataset_id'],
            'test_sha256': manifest['test_checksum'], 'test_image_sha256':manifest['image_checksum'],
            'configuration': config, 'metrics': metrics, 'predictions_path': str(prediction_path),
            'size_stratified':{'status':'measured','bins_native_pixel_area':{'small':[0,1024],'medium':[1024,9216],'large':[9216,None]},
                'counts':{k:dict(v) for k,v in size_totals.items()},'small_object_ap':None,
                'method':'Class-aware operating-point recall by ground-truth native box area; unmatched false positives stratified separately by predicted area. Descriptive bins, not detectability thresholds or official COCO area AP.'},
            'test_images': len(images), 'settings': {'size_evaluation_version':'native_area_recall_1.0','ap_confidence_floor': .001, 'operating_confidence': config['confidence'],
            'nms_iou': config['iou'], 'matching_iou': config['match_iou'], 'image_size': payload['configuration']['image_size'],
            'max_detections': 300, 'device': selected_device,'ground_truth_protocol_version':'canonical_boxes_1.0'}, 'per_class': per_class, 'curves': curves,
            'confusion_matrix':{'labels':classes+['background'],'counts':confusion.tolist(),
                'method':'Rows predicted, columns true; confidence-ordered spatial IoU matching ignoring class. Off-diagonal entries show class confusion; background records unmatched predictions/truth. Primary precision/recall use separate class-aware matching.'},
            'ground_truth_protocol':{'format':'Canonical normalized YOLO boxes with complete one-to-one class mapping',
                'source_format':store.get(config['dataset_id'],'dataset')['payload']['profile'].get('annotation_format'),
                'crowd_policy':'All retained parsed objects are evaluated as ordinary boxes. This adapter does not reproduce official COCO crowd-ignore semantics; published COCO benchmarks are not assumed comparable.'},
            'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'compute': compute()},
            'error_denominators': {'false_positives_per_image': 'false detections / evaluated images; not a true-negative-based false-positive rate',
                'false_negative_fraction': 'missed labeled objects / labeled objects',
                'inference_latency_ms': 'Mean inference time across evaluated images; excludes loading, preprocessing and NMS'},
            'notes': ['Precision/recall/F1 use class-aware confidence-ordered matching at the declared operating threshold.',
            'mAP uses Ultralytics confidence sweep; latency is model inference time, excludes loading/preprocessing/NMS, and is descriptive.']})
        results.append(evaluation['id'])
    return {'evaluation_ids': results}
