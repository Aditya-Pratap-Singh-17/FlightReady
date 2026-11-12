"""Inference-only adapters and established COCO bbox evaluation on local labels."""
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import time
import numpy as np
from PIL import Image
from backend.context import DatasetContext
from backend.compatibility import compatibility
from backend.workbench.datasets import load, image_bytes, training_eligibility
from backend.workbench.detector import detection_errors, native_size_outcomes, test_manifest
from backend.workbench.model_registry import verify_available, ARCHITECTURES


class DetectionAdapter(Protocol):
    def predict(self, image: Image.Image) -> list[dict]: ...


class YolosAdapter:
    def __init__(self, payload, device, image_size):
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ValueError('Install requirements/requirements-inference.txt for safe model inference') from exc
        if not verify_available(payload):
            raise ValueError('Cached model is missing or has changed. Reacquire the pinned model.')
        self.torch = torch
        self.device = 'cuda:0' if device == '0' else 'cpu'
        if self.device != 'cpu' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; choose CPU or auto')
        if device == 'auto' and torch.cuda.is_available(): self.device = 'cuda:0'
        # Concrete built-in classes only; no repository-defined Python classes.
        config=json.loads((Path(payload['directory'])/'config.json').read_text(encoding='utf-8'))
        self.model_type=config['model_type']
        if self.model_type not in ARCHITECTURES: raise ValueError('Unsupported built-in detection architecture')
        model_name,processor_name=ARCHITECTURES[self.model_type]
        model_class,processor_class=getattr(transformers,model_name),getattr(transformers,processor_name)
        self.processor = processor_class.from_pretrained(payload['directory'], local_files_only=True)
        model_config=model_class.config_class.from_pretrained(payload['directory'],local_files_only=True)
        # Full detector weights already contain the backbone; never acquire a
        # second model behind the user's back while constructing DETR/RT-DETR.
        if hasattr(model_config,'use_pretrained_backbone'): model_config.use_pretrained_backbone=False
        try:
            self.model = model_class.from_pretrained(payload['directory'], config=model_config,local_files_only=True,
                use_safetensors=True, trust_remote_code=False).to(self.device).eval()
        except ImportError as exc:
            raise ValueError('A detector dependency is missing. Install requirements/requirements-inference.txt, including timm for DETR backbones.') from exc
        self.image_size = image_size

    def predict(self, image):
        torch = self.torch
        size={'height':self.image_size,'width':self.image_size} if self.model_type=='rt_detr' else {'shortest_edge': self.image_size, 'longest_edge': 2*self.image_size}
        inputs = self.processor(images=image, return_tensors='pt',size=size)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        data = self.processor.post_process_object_detection(outputs, threshold=0., target_sizes=torch.tensor([[image.height, image.width]], device=self.device))[0]
        predicted = []
        for score, label, box in zip(data['scores'], data['labels'], data['boxes']):
            name = self.model.config.id2label[int(label)]
            if name == 'N/A': continue
            x1, y1, x2, y2 = box.cpu().tolist()
            predicted.append({'class': name, 'confidence': float(score),
                'bbox': [(x1+x2)/2/image.width, (y1+y2)/2/image.height, (x2-x1)/image.width, (y2-y1)/image.height]})
        return sorted(predicted, key=lambda p: p['confidence'], reverse=True)[:100]


def validate_benchmark(store, config):
    ds = load(store, config['dataset_id'])
    dataset = store.get(config['dataset_id'], 'dataset')['payload']
    if ds.modality!='image': raise ValueError('Select an image dataset for detector inference')
    acquired = store.get(config['acquired_model_id'], 'acquired_model')['payload']
    if acquired.get('adapter') not in ('transformers_yolos','transformers_detection'):
        raise ValueError('Unsupported inference adapter')
    if not verify_available(acquired): raise ValueError('Cached model integrity check failed; reacquire the model')
    if config.get('mode')=='preview':
        from backend.workbench.splits import get_split
        names=get_split(store,config['split_id'],config['dataset_id'])['images']['test'] if config.get('split_id') else sorted(ds.image_hashes)
        readable=set(ds.frame.loc[ds.frame.readable,'image'])
        names=[n for n in names if n in readable]
        if config.get('max_images'): names=names[:config['max_images']]
        if not names: raise ValueError('The selected population contains no readable images')
        digest=hashlib.sha256(json.dumps([(n,ds.image_hashes[n]) for n in names]).encode()).hexdigest()
        return ds,acquired,{}, {'test_checksum':digest,'image_checksum':digest},names
    eligibility=training_eligibility(ds)
    if not eligibility['eligible']:
        raise ValueError('Reliable detection ground truth is required for metrics: ' + '; '.join(eligibility['reasons']) + ' Use prediction preview to inspect images without accuracy claims.')
    if config.get('max_images'): raise ValueError('Image limits are supported for prediction previews only; benchmark the declared evaluation partition for metrics')
    available_labels=set(ds.profile.get('declared_classes') or ds.profile['class_counts'])
    labels=set(config.get('evaluation_classes') or available_labels)
    if not labels or not labels<=available_labels: raise ValueError('Select annotated dataset classes for the evaluation scope')
    mapping = config.get('class_mapping') or {name: name for name in labels}
    if set(mapping) != labels or not set(mapping.values()) <= set(acquired['classes']) or len(set(mapping.values())) != len(mapping):
        raise ValueError('Every dataset class requires a complete one-to-one mapping into model classes. Drone targets are not COCO airplane or bird classes.')
    if any(c.casefold() in ('drone','drones','uav','uavs') and c not in acquired['classes'] for c in labels):
        raise ValueError('This pretrained model has no drone-target class; a class mapping cannot turn COCO birds/airplanes into drone targets.')
    if config.get('readiness_id'):
        ready = store.get(config['readiness_id'], 'job')['payload']
        if ready.get('status') != 'completed' or ready.get('task') != 'readiness' or ready['config']['dataset_id'] != config['dataset_id']:
            raise ValueError('Choose a completed assessment of this exact dataset version')
    if config.get('validate_project_goal',True):
        if not config.get('readiness_id'): raise ValueError('Select a completed assessment to validate its goal, or choose annotation-only benchmarking')
        context = DatasetContext.model_validate(ready['result']['confirmed_context'])
        if context.goal.task_type != 'object_detection': raise ValueError('This adapter evaluates object detection only')
        if not set(context.goal.required_classes)<=labels: raise ValueError('Goal validation must include every required target class; use annotation-only benchmarking for a restricted class scope')
        preview = compatibility(ds, context)
        if preview['status'] != 'supported': raise ValueError(preview['summary'])
    manifest, names = test_manifest(store, config)
    if not names or not any(o['image'] in names and o['class'] in labels for o in ds.objects):
        raise ValueError('Evaluation partition needs annotated objects to calculate AP/recall')
    if not config.get('split_id'):
        role=ds.profile.get('source_split')
        if not role and 'split' in ds.frame:
            roles=set(ds.frame['split'].dropna().astype(str).str.lower()); role=next(iter(roles)) if len(roles)==1 else None
        if role not in ('val','test','validation','testing') and not config.get('acknowledge_evaluation_scope'):
            raise ValueError('Select a held-out split or acknowledge the entire labeled dataset is an inference benchmark, not an independent test')
    return ds, acquired, mapping, manifest, names


def coco_metrics(samples, classes):
    """Official pycocotools bbox conventions, ordinary canonical boxes, maxDets=100."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ValueError('Install requirements/requirements-inference.txt for COCO evaluation') from exc
    def pixel_box(box, w, h):
        x,y,bw,bh = box
        return [(x-bw/2)*w, (y-bh/2)*h, bw*w, bh*h]
    data = {'images': [], 'annotations': [], 'categories': [{'id': i+1, 'name': name} for i,name in enumerate(classes)]}
    predictions = []
    for i,sample in enumerate(samples, 1):
        w,h = sample['width'], sample['height']
        data['images'].append({'id': i, 'width': w, 'height': h})
        for obj in sample['ground_truth']:
            box = pixel_box(obj['bbox'],w,h)
            data['annotations'].append({'id': len(data['annotations'])+1, 'image_id': i,
                'category_id': classes.index(obj['class'])+1, 'bbox': box, 'area': box[2]*box[3], 'iscrowd': 0})
        for obj in sample['all_predictions']:
            if obj['class'] in classes:
                predictions.append({'image_id': i, 'category_id': classes.index(obj['class'])+1,
                    'bbox': pixel_box(obj['bbox'],w,h), 'score': obj['confidence']})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = data; gt.createIndex()
        if predictions:
            pred = gt.loadRes(predictions)
        else:
            pred = COCO(); pred.dataset = {'images':data['images'], 'categories':data['categories'], 'annotations':[]}; pred.createIndex()
        evaluator = COCOeval(gt,pred,'bbox'); evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    def metric(i): return float(evaluator.stats[i]) if evaluator.stats[i] >= 0 else None
    per_class = {}
    for k,name in enumerate(classes):
        values = evaluator.eval['precision'][:,:,k,0,-1]; values = values[values >= 0]
        per_class[name] = float(values.mean()) if values.size else None
    return {'mAP50_95':metric(0), 'mAP50':metric(1), 'AP_small':metric(3), 'AP_medium':metric(4), 'AP_large':metric(5),
            'per_class_mAP50_95':per_class}


def benchmark(store, job_id, config):
    ds, acquired, mapping, manifest, names = validate_benchmark(store, config)
    adapter = YolosAdapter(acquired, config['device'], config['image_size'])
    samples = []; totals = Counter(); sizes = defaultdict(Counter); latencies = []
    by_image = defaultdict(list)
    preview=config.get('mode')=='preview'
    for obj in ds.objects: by_image[obj['image']].append(obj)
    for i,name in enumerate(names, 1):
        raw = image_bytes(store, config['dataset_id'], name)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGB')
        start = time.perf_counter(); predicted = adapter.predict(image); latencies.append(1000*(time.perf_counter()-start))
        truth = [{**o, 'class':mapping.get(o['class'],o['class'])} for o in by_image[name] if preview or o['class'] in mapping]
        # Evaluate the declared label subset. Unrelated model classes are explicitly out of scope.
        selected = predicted if preview else [p for p in predicted if p['class'] in mapping.values()]
        operating = [p for p in selected if p['confidence'] >= config['confidence']]
        if preview:
            annotated=[{**p,'outcome':'unscored_prediction'} for p in operating]; missed=[]
        else:
            errors, annotated, missed = detection_errors(truth, operating, config['match_iou'])
            totals.update({k:errors[k] for k in ('true_positive','false_positive','false_negative')})
            for key,values in native_size_outcomes(truth,annotated,missed,image.width,image.height).items():
                sizes[key].update({k:v for k,v in values.items() if k != 'recall'})
        samples.append({'image':name, 'width':image.width, 'height':image.height, 'ground_truth':truth,
                        'predictions':annotated, 'all_predictions':selected, 'missed':missed,'scored':not preview})
        store.update(job_id, stage=f'Real inference {i}/{len(names)}', progress={'images':i,'total_images':len(names)})
    metrics = {} if preview else coco_metrics(samples, sorted(mapping.values()))
    tp,fp,fn = (totals[k] for k in ('true_positive','false_positive','false_negative'))
    if not preview:
        metrics.update(dict(totals)); metrics.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None)
    metrics.update(mean_pipeline_latency_ms=float(np.mean(latencies)), latency_images=len(latencies))
    path = store.directory('jobs',job_id) / 'predictions.json'; path.write_text(json.dumps(samples,allow_nan=False),encoding='utf-8')
    versions = {name:importlib.metadata.version(name) for name in ('torch','transformers','pycocotools','safetensors')}
    record = store.create('benchmark', {'job_id':job_id, 'kind':'inference_preview' if preview else 'inference_only', 'readiness_id':config.get('readiness_id'),
        'dataset_id':config['dataset_id'], 'dataset_sha256':ds.sha256, 'test_sha256':manifest['test_checksum'], 'test_images':len(names),
        'acquired_model_id':config['acquired_model_id'], 'model':acquired, 'configuration':config, 'metrics':metrics,
        'predictions_path':str(path), 'size_stratified':{k:dict(v) for k,v in sizes.items()},
        'reproducibility':{'dependencies':versions,'python':platform.python_version(),'metric_version':'pycocotools-canonical-bbox-1.0',
            'adapter_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'evaluation_scope':{'scored':not preview,'classes':sorted(mapping),'excluded_classes':sorted(set(ds.profile.get('class_counts',{}))-set(mapping)),
            'project_goal_validated':not preview and config.get('validate_project_goal',True)},
        'limitations':(['Prediction preview only: no AP, precision, recall or false-negative claims are calculated.'] if preview else [])+[
            'Only explicitly selected annotation classes are benchmarked. Excluded classes and unmet project requirements are not validated.',
            'Inference-only benchmarking does not test a data-quality intervention.',
            'Pretrained development-data overlap with this evaluation set is unverified.',
            'AP uses COCO IoU .50:.05:.95, 101 recall positions and maxDets 100. Canonical boxes do not preserve official crowd/ignore semantics.',
            'Only explicitly mapped label classes are scored; predictions in other COCO classes are out of scope.',
            'Latency includes preprocessing, model execution and postprocessing; it is distinct from model-only inference time.',
            'Nighttime subset performance requires recorded verified condition labels; it is not inferred from image brightness.']})
    return {'benchmark_id':record['id']}
