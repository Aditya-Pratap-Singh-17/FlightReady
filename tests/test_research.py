"""Train-only interventions and export invariants on tiny known YOLO fixtures."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
import pytest
from PIL import Image
from backend.context import DatasetContext,Goal
from backend.research.variants import VariantConfig,create_variant,_file_sha
from backend.research.experiments import assess_variant,export_experiments,train_detector,TrainingConfig

@pytest.fixture
def parent(tmp_path):
    target=tmp_path/'parent.zip'
    with zipfile.ZipFile(target,'w') as archive:
        archive.writestr('classes.txt','car\npedestrian\n')
        metadata=['image,split,flight,weather,viewpoint']
        for i in range(6):
            split='train' if i<4 else 'val' if i==4 else 'test'
            image=f'images/{split}/{i}.png';label=f'labels/{split}/{i}.txt'
            raw=io.BytesIO();Image.new('RGB',(16,16),color=(i*30,40,40)).save(raw,format='PNG')
            archive.writestr(image,raw.getvalue());archive.writestr(label,f'{i%2} .5 .5 .2 .2\n')
            metadata.append(f'{image},{split},flight{i},{"clear" if i%2==0 else "rainy"},{"nadir" if i%2==0 else "oblique"}')
        archive.writestr('metadata.csv','\n'.join(metadata)+'\n')
    return target


@pytest.mark.parametrize('parameters',[{'noise_sigma':20},{'brightness_factor':.6},{'blur_radius':2}])
def test_new_pixel_interventions_reproducible_and_test_immutable(parent,tmp_path,parameters):
    cfg=VariantConfig(seed=81,intervention='image_degradation',parameters=parameters,group_column='flight')
    a=create_variant(parent,tmp_path/'new-a',cfg);b=create_variant(parent,tmp_path/'new-b',cfg)
    assert a==b
    assert a['implementation_version']=='interventions-2.0'
    assert a['software']['numpy'] and all(m['parent_object_count']==m['output_object_count'] for m in a['modifications'])
    with zipfile.ZipFile(parent) as original,zipfile.ZipFile(tmp_path/'new-a'/'dataset.zip') as variant:
        for name in original.namelist():
            if '/val/' in name or '/test/' in name:assert original.read(name)==variant.read(name)


def test_small_object_reduction_is_whole_image_train_only(parent,tmp_path):
    cfg=VariantConfig(seed=2,intervention='small_object_reduction',parameters={'maximum_native_area':16,'retain_fraction':.5,'classes':['car']},group_column='flight')
    manifest=create_variant(parent,tmp_path/'small',cfg)
    assert manifest['training_images']==3 and len(manifest['removed_training_images'])==1
    assert all('/train/' in name for name in manifest['removed_training_images'])


def test_hard_negative_reduction_requires_explicit_metadata_and_preserves_test(parent,tmp_path):
    rewritten=tmp_path/'negatives.zip'
    with zipfile.ZipFile(parent) as source,zipfile.ZipFile(rewritten,'w') as output:
        for name in source.namelist():
            raw=source.read(name)
            if name=='metadata.csv':
                lines=raw.decode().splitlines();raw=('\n'.join([lines[0]+',candidate']+[line+',true' for line in lines[1:]])+'\n').encode()
            if name.startswith('labels/train/') and name.endswith(('0.txt','1.txt')):raw=b''
            output.writestr(name,raw)
    config=VariantConfig(seed=2,intervention='negative_coverage',parameters={'column':'candidate','retain_fraction':.5,'classes':['car','pedestrian']},group_column='flight')
    manifest=create_variant(rewritten,tmp_path/'hard',config)
    assert manifest['training_images']==3
    with zipfile.ZipFile(rewritten) as original,zipfile.ZipFile(tmp_path/'hard'/'dataset.zip') as variant:
        assert original.read('labels/test/5.txt')==variant.read('labels/test/5.txt')
    config.parameters['column']='missing'
    with pytest.raises(ValueError,match='metadata'):create_variant(rewritten,tmp_path/'missing-hard',config)

@pytest.mark.parametrize('intervention,parameters,target',[
    ('control',{},4),('annotation_noise',{'probability':1},4),
    ('object_scale',{'relative_area_range':[0,.5]},4),
    ('redundancy',{'unique_images':1},4),('class_representation',{'classes':['car']},2),
    ('environment',{'column':'weather','allowed_values':['clear']},2),
    ('viewpoint',{'column':'viewpoint','allowed_values':['nadir']},2),
    ('shortcut',{'column':'weather','class_condition_map':{'car':'clear','pedestrian':'rainy'}},4),
    ('image_degradation',{'blur_radius':1,'brightness_factor':.5},4)])
def test_variants_reproducible_immutable_evaluation(parent,tmp_path,intervention,parameters,target):
    before=_file_sha(parent)
    config=VariantConfig(seed=7,intervention=intervention,parameters=parameters,target_train_images=target,group_column='flight')
    a=create_variant(parent,tmp_path/'a',config);b=create_variant(parent,tmp_path/'b',config)
    assert a==b and a['variant_id']==b['variant_id'] and a['parent_dataset_id']==before
    assert a['training_images']==target and _file_sha(parent)==before
    with zipfile.ZipFile(parent) as original,zipfile.ZipFile(tmp_path/'a'/'dataset.zip') as changed:
        for name in original.namelist():
            if '/val/' in name or '/test/' in name:
                assert original.read(name)==changed.read(name)
                assert hashlib.sha256(original.read(name)).hexdigest()==a['evaluation_sha256'][name]
        if intervention=='annotation_noise':
            assert original.read('labels/train/0.txt')!=changed.read('labels/train/0.txt')
        if intervention=='redundancy':assert a['unique_training_source_images']==1
    with pytest.raises(ValueError,match='already exists'):create_variant(parent,tmp_path/'a',config)

def test_group_and_content_overlap_rejected_before_writing(parent,tmp_path):
    raw=tmp_path/'overlap.zip'
    with zipfile.ZipFile(parent) as source,zipfile.ZipFile(raw,'w') as archive:
        for name in source.namelist():
            data=source.read(name)
            if name=='metadata.csv':data=data.replace(b'flight4',b'flight0')
            archive.writestr(name,data)
    config=VariantConfig(seed=1,intervention='control',group_column='flight')
    with pytest.raises(ValueError,match='multiple splits'):create_variant(raw,tmp_path/'blocked',config)
    assert not (tmp_path/'blocked').exists()
    with pytest.raises(ValueError,match='acknowledge'):create_variant(parent,tmp_path/'missing-groups',VariantConfig(seed=1,intervention='control'))
    with zipfile.ZipFile(parent) as source,zipfile.ZipFile(tmp_path/'same.zip','w') as archive:
        for name in source.namelist():
            archive.writestr(name,source.read('images/train/0.png') if name=='images/test/5.png' else source.read(name))
    with pytest.raises(ValueError,match='content overlaps'):create_variant(tmp_path/'same.zip',tmp_path/'blocked2',config)

def test_feature_export_has_no_invented_training_results(parent,tmp_path):
    directory=tmp_path/'run';create_variant(parent,directory,VariantConfig(seed=1,intervention='control',target_train_images=2,group_column='flight'))
    ctx=DatasetContext(original_text='confirmed synthetic',confirmed=True,goal=Goal(task_type='object_detection',required_classes=['car'],split_column='split',group_column='flight'))
    payload=assess_variant(directory,ctx)
    count=next(m for m in payload['training_features'] if m['metric_id']=='row_count')
    assert count['value']==2
    assert payload['assessment']['dataset']['profile']['image_count']==4
    rows=export_experiments([directory],tmp_path/'experiments.csv')
    assert len(rows)==2 and all(r['mAP50'] is None for r in rows)
    with (tmp_path/'experiments.csv').open() as source:parsed=list(csv.DictReader(source))
    assert parsed[0]['mAP50']=='' and json.loads(parsed[0]['readiness_features'])
    with pytest.raises(FileExistsError):export_experiments([directory],tmp_path/'experiments.csv')

def test_training_requires_separate_explicit_invocation(tmp_path):
    cfg=TrainingConfig(model_version='user-supplied',package_version='user-pinned',checkpoint='missing.pt',
        checkpoint_sha256='none',seed=0,epochs=1,batch=1,image_size=64,learning_rate=.01)
    with pytest.raises(ValueError,match='explicit'):train_detector(tmp_path,cfg)
    with pytest.raises(ValueError,match='trusted local'):train_detector(tmp_path,cfg,execute=True)

def test_matched_size_cannot_silently_duplicate(parent,tmp_path):
    config=VariantConfig(seed=0,intervention='environment',parameters={'column':'weather','allowed_values':['clear']},target_train_images=3,group_column='flight')
    with pytest.raises(ValueError,match='implicit duplication'):create_variant(parent,tmp_path/'oversized',config)


def test_optional_detector_orchestration_with_test_double(parent,tmp_path,monkeypatch):
    """Known fake outputs exercise export plumbing, never a detector benchmark."""
    import sys
    import types
    import backend.research.experiments as experiments
    directory=tmp_path/'mock-run';create_variant(parent,directory,VariantConfig(seed=3,intervention='control',group_column='flight'))
    (directory/'readiness-features.json').write_text('{}')
    checkpoint=tmp_path/'test.pt';checkpoint.write_bytes(b'mocked checkpoint')
    calls=[]
    class Detector:
        def __init__(self,path,**kwargs):assert Path(path).is_file()
        def train(self,**kwargs):
            calls.append(kwargs);weights=Path(kwargs['project'])/kwargs['name']/'weights';weights.mkdir(parents=True)
            (weights/'best.pt').write_bytes(b'mocked trained checkpoint')
        def val(self,**kwargs):
            return types.SimpleNamespace(box=types.SimpleNamespace(map50=.5,map=.25,mp=.4,mr=.3,ap_class_index=[0],ap=[.25]))
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=Detector,settings={}))
    monkeypatch.setattr(experiments.importlib.metadata,'version',lambda name:'test-version')
    cfg=TrainingConfig(model_version='test-double',package_version='test-version',checkpoint=str(checkpoint),
        checkpoint_sha256=_file_sha(checkpoint),seed=3,epochs=1,batch=2,image_size=64,learning_rate=.01,
        condition_subsets={'declared-condition':['images/test/5.png']})
    result=train_detector(directory,cfg,execute=True)
    assert result['results']['test']['mAP50']==.5
    assert result['results']['val']['per_class_mAP50_95']=={'0':.25}
    assert result['condition_specific_performance']['declared-condition']['recall']==.3
    assert calls[0]['amp'] is False and calls[0]['plots'] is False and calls[0]['seed']==3
    rows=export_experiments([directory],tmp_path/'mock-only.csv')
    assert rows[0]['mAP50']==.5 and 'test-double' in rows[0]['training_configuration']
