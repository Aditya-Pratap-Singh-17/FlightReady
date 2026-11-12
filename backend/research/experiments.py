"""Configuration, assessment-feature export, and explicitly invoked local training."""
from pathlib import Path
import csv
import io
import json
import os
import platform
import zipfile
import importlib.metadata
import hashlib
from backend.context import Model,DatasetContext
from pydantic import Field
from backend.research.variants import _file_sha,VariantConfig,assign_split

class TrainingConfig(Model):
    model_family: str = 'ultralytics-yolo'
    model_version: str
    package_version: str
    checkpoint: str
    checkpoint_sha256: str
    seed: int
    epochs: int = Field(ge=1)
    batch: int = Field(ge=1)
    image_size: int = Field(ge=32,le=4096)
    device: str = 'cpu'
    workers: int = Field(default=0,ge=0,le=16)
    optimizer: str = 'SGD'
    learning_rate: float = Field(gt=0)
    # Explicit condition image lists are assessed with unchanged annotations.
    # Producing these lists is a separate independent sampling/audit activity.
    condition_subsets: dict[str,list[str]] = Field(default_factory=dict)

def assess_variant(directory,context:DatasetContext):
    from backend.large_ingestion import load_dataset_stream
    from backend.readiness_engine import assess
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if _file_sha(directory/'dataset.zip')!=manifest['variant_id']:raise ValueError('Variant archive hash mismatch')
    with (directory/'dataset.zip').open('rb') as source:ds=load_dataset_stream('dataset.zip',source)
    try:
        cert=assess(ds,context)
        # Export train-only features as well as the complete collection report.
        # This prevents unchanged evaluation partitions masking interventions.
        split=VariantConfig.model_validate(manifest['config']).split_column
        if split not in ds.frame:raise ValueError('Variant split column missing')
        import copy
        train=copy.copy(ds);train.frame=ds.frame[ds.frame[split].eq('train')].copy()
        names=set(train.frame.image);train.objects=[o for o in ds.objects if o['image'] in names]
        train.sha256=hashlib.sha256((ds.sha256+json.dumps(sorted(names))).encode()).hexdigest()
        train.issues=[i for i in ds.issues if i.get('reference') in names]
        train.profile={**ds.profile,'image_count':len(names),'object_count':len(train.objects),'issues':train.issues,
            'assessment_scope':'training image subset','parent_archive_sha256':ds.sha256}
        train.image_hashes={k:v for k,v in ds.image_hashes.items() if k in names}
        train.near_pairs=[(a,b) for a,b in ds.near_pairs if a in names and b in names]
        train_context=context.model_copy(deep=True)
        train_cert=assess(train,train_context)
        payload={'variant_id':manifest['variant_id'],'assessment':cert.model_dump(mode='json'),
            'training_features':[m.model_dump(mode='json') for m in train_cert.metrics],
            'feature_scope':'training images only; full assessment includes immutable validation/test partitions'}
        target=directory/'readiness-features.json'
        with target.open('x',encoding='utf-8') as destination:json.dump(payload,destination,indent=2,allow_nan=False)
        return payload
    finally:ds.close()

def _metrics(result):
    box=result.box
    # ap_class_index excludes absent classes; box.maps can contain overall mAP
    # placeholders for absent classes, so never export it as measured per-class AP.
    classes=getattr(box,'ap_class_index',[])
    maps=getattr(box,'ap',[])
    return {'mAP50':float(box.map50),'mAP50_95':float(box.map),
            'precision':float(box.mp),'recall':float(box.mr),
            'per_class_mAP50_95':{str(int(c)):float(v) for c,v in zip(classes,maps)}}

def train_detector(directory,config:TrainingConfig,*,execute=False):
    """No training or imports unless caller explicitly sets execute=True."""
    if not execute:raise ValueError('Training requires explicit execute=True / --execute')
    if config.model_family!='ultralytics-yolo':raise ValueError('Only local Ultralytics detection is currently supported')
    directory=Path(directory).resolve();checkpoint=Path(config.checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower()!='.pt' or _file_sha(checkpoint)!=config.checkpoint_sha256:
        raise ValueError('A trusted local .pt checkpoint with matching SHA256 is required; no downloads')
    try:installed=importlib.metadata.version('ultralytics')
    except importlib.metadata.PackageNotFoundError as exc:raise ValueError('Optional ultralytics package is not installed; configure it separately before training') from exc
    if installed!=config.package_version:raise ValueError('Installed detector package version differs from pinned experiment configuration')
    manifest=json.loads((directory/'manifest.json').read_text())
    archive_path=directory/'dataset.zip'
    if _file_sha(archive_path)!=manifest['variant_id']:raise ValueError('Variant hash mismatch')
    if not (directory/'readiness-features.json').is_file():raise ValueError('Assess the variant before training')
    root=directory/'training-data'
    if root.exists() or (directory/'run').exists() or (directory/'experiment.json').exists():raise ValueError('Training artifacts already exist; use a new experiment directory')
    from backend.image_ingestion import validate_members
    import pandas as pd
    import yaml
    with zipfile.ZipFile(archive_path) as source:
        names,_=validate_members(source)
        meta=pd.read_csv(io.BytesIO(source.read('metadata.csv'))).set_index('image').to_dict('index')
        cfg=VariantConfig.model_validate(manifest['config']);splits={key:[] for key in ('train','val','test')}
        from backend.research.variants import label_path
        members=set(meta)|{label_path(image) for image in meta}
        root.mkdir()
        for name in sorted(names):
            if name not in members or name.endswith('/'):continue
            # Only validated image/annotation members are materialized for an
            # explicitly requested detector run, never scripts from an upload.
            suffix=Path(name).suffix.lower()
            if suffix not in ('.png','.jpg','.jpeg','.webp','.bmp','.txt'):continue
            target=root.joinpath(*Path(name).parts).resolve()
            if not target.is_relative_to(root):raise ValueError('Unsafe training member')
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open(name) as member,target.open('xb') as output:
                while chunk:=member.read(1024**2):output.write(chunk)
            if name in meta:
                split=assign_split(name,meta.get(name,{}),cfg);splits[split].append(name)
        data={'path':str(root),'names':source.read('classes.txt').decode('utf-8-sig').splitlines()}
    for split,images in splits.items():
        listing=root/(split+'.txt');listing.write_text('\n'.join('./'+im for im in images)+'\n',encoding='utf-8');data[split]=str(listing)
    data_file=root/'data.yaml';data_file.write_text(yaml.safe_dump(data),encoding='utf-8')
    # Offline intent plus local paths; no remote checkpoint/data resolution.
    os.environ['YOLO_OFFLINE']='true'
    from ultralytics import YOLO,settings
    settings.update({'sync':False})
    model=YOLO(str(checkpoint),task='detect')
    model.train(data=str(data_file),epochs=config.epochs,batch=config.batch,imgsz=config.image_size,
        seed=config.seed,deterministic=True,device=config.device,workers=config.workers,optimizer=config.optimizer,
        lr0=config.learning_rate,project=str(directory),name='run',exist_ok=False,plots=False,amp=False,
        patience=0,pretrained=False,cache=False)
    trained=directory/'run'/'weights'/'best.pt'
    if not trained.is_file():raise ValueError('Detector did not produce the expected best checkpoint')
    model=YOLO(str(trained),task='detect')
    results={split:_metrics(model.val(data=str(data_file),split=split,device=config.device,plots=False)) for split in ('val','test')}
    conditions={}
    for name,images in config.condition_subsets.items():
        if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):raise ValueError('Invalid condition name')
        if not images or not set(images).issubset(set(splits['test'])):raise ValueError('Condition subsets must contain only unchanged test images')
        listing=root/('condition-'+name+'.txt');listing.write_text('\n'.join('./'+im for im in images)+'\n',encoding='utf-8')
        condition_data={**data,'test':str(listing)};path=root/('condition-'+name+'.yaml');path.write_text(yaml.safe_dump(condition_data),encoding='utf-8')
        conditions[name]=_metrics(model.val(data=str(path),split='test',device=config.device,plots=False))
    for member,expected_hash in manifest['evaluation_sha256'].items():
        if _file_sha(root/member)!=expected_hash:raise ValueError('Evaluation data changed during training; results cannot be recorded')
    if _file_sha(archive_path)!=manifest['variant_id']:raise ValueError('Variant archive changed during training')
    payload={'variant_id':manifest['variant_id'],'parent_dataset_id':manifest['parent_dataset_id'],
        'modified_property':manifest['modified_property'],'intervention':manifest['config'],
        'training_configuration':config.model_dump(exclude={'checkpoint'}),'model_sha256':_file_sha(trained),
        'seed':config.seed,'results':results,'condition_specific_performance':conditions,
        'environment':{'python':platform.python_version(),'platform':platform.platform(),
            'packages':{n:importlib.metadata.version(n) for n in ('ultralytics','torch','numpy')}}}
    encoded=json.dumps(payload,indent=2,allow_nan=False)
    with (directory/'experiment.json').open('x',encoding='utf-8') as output:output.write(encoded)
    return payload

def export_experiments(directories,destination):
    """One row per variant/evaluation split; absent performance stays null."""
    rows=[]
    for directory in map(Path,directories):
        manifest=json.loads((directory/'manifest.json').read_text());result_path=directory/'experiment.json'
        result=json.loads(result_path.read_text()) if result_path.is_file() else {}
        feature_path=directory/'readiness-features.json';features=json.loads(feature_path.read_text()) if feature_path.is_file() else {}
        if result and result['variant_id']!=manifest['variant_id']:raise ValueError('Experiment and manifest mismatch')
        for split in ('val','test'):
            row={'variant_id':manifest['variant_id'],'parent_dataset_id':manifest['parent_dataset_id'],
                'modified_property':manifest['modified_property'],'seed':result.get('seed',manifest['config']['seed']),
                'training_images':manifest['training_images'],'unique_training_source_images':manifest['unique_training_source_images'],
                'split':split,'mAP50':None,'mAP50_95':None,'precision':None,'recall':None}
            row.update({k:v for k,v in result.get('results',{}).get(split,{}).items() if k in row})
            for key,value in {'intervention':manifest['config'],'training_configuration':result.get('training_configuration'),
                'readiness_features':features.get('training_features'),'per_class':result.get('results',{}).get(split,{}).get('per_class_mAP50_95'),
                'condition_performance':result.get('condition_specific_performance'),'environment':result.get('environment')}.items():
                row[key]=json.dumps(value,sort_keys=True,allow_nan=False)
            rows.append(row)
    if not rows:raise ValueError('No experiment directories supplied')
    with Path(destination).open('x',encoding='utf-8',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows
