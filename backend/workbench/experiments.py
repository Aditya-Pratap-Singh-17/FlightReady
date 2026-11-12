"""Named, immutable protocols executed in the existing isolated compute worker."""
import copy
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from backend.context import Model, DatasetContext
from metrics.quality_profiles import fingerprint
from backend.workbench.contracts import TrainRequest, TestRequest
from backend.workbench.detector import validate_training, classes_for, class_mapping, test_manifest, validation_source, training_only_manifest
from backend.workbench.splits import get_split, assert_independent


class ExperimentRequest(Model):
    name: str = Field(min_length=1, max_length=160)
    hypothesis: str = Field(default='', max_length=4000)
    candidate: TrainRequest
    reference: TrainRequest | None = None
    modes: list[Literal['candidate', 'reference', 'combined', 'controlled']] = Field(default_factory=lambda: ['candidate', 'reference'], min_length=1, max_length=4)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=20)
    test_dataset_id: str
    test_split_id: str | None = None
    test_class_mapping: dict[str, str] = Field(default_factory=dict)
    reference_class_mapping: dict[str, str] = Field(default_factory=dict)
    size_mode: Literal['natural', 'matched'] = 'natural'
    selection_seed: int = Field(default=42, ge=0)
    confidence: float = Field(default=.25, gt=0, lt=1)
    iou: float = Field(default=.7, gt=0, lt=1)
    match_iou: float = Field(default=.5, gt=0, lt=1)
    acknowledge_missing_groups: bool = False

    @model_validator(mode='after')
    def valid(self):
        if len(set(self.modes)) != len(self.modes) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Modes and random seeds must be unique')
        if any(isinstance(s, bool) or s < 0 or s > 2**31-1 for s in self.seeds):
            raise ValueError('Seeds must be nonnegative 32-bit integers')
        if any(m in self.modes for m in ('reference', 'combined')) and self.reference is None:
            raise ValueError('Reference and combined modes require a reference training configuration')
        if self.size_mode=='matched' and self.reference is None:
            raise ValueError('Matched-size protocols require a reference dataset to define the comparison count')
        if 'combined' in self.modes and self.size_mode == 'matched':
            raise ValueError('Combined training intentionally changes sample count; select natural sizes or run a separate matched-size protocol')
        return self


def validate_experiment(store, request):
    configurations = [request.candidate] + ([request.reference] if request.reference else [])
    manifests = []
    canonical,_ = validate_training(store,request.candidate.model_dump())
    sizes=[]
    for i, config in enumerate(configurations):
        effective=config.model_dump()
        if i:
            effective.update({k:getattr(request.candidate,k) for k in ('architecture','initialization','weights_path','epochs','batch','image_size','learning_rate','device')})
            effective['class_mapping']=request.reference_class_mapping
        classes, manifest = validate_training(store, effective)
        sizes.append(len(manifest['images']['train']))
        readiness = store.get(config.readiness_id, 'job')['payload']['result']
        if readiness['confirmed_context']['goal']['task_type'] != 'object_detection':
            raise ValueError('The validation lab requires detection-task readiness evidence')
        if i:
            class_mapping(classes_for(store, config.dataset_id), canonical, request.reference_class_mapping)
            if config.dataset_id == request.candidate.dataset_id:
                raise ValueError('Reference training must use a separate dataset snapshot')
        validation=validation_source(store,effective,classes)
        manifests.append(training_only_manifest(manifest) if validation else manifest)
        if validation:
            manifests.append(validation['manifest'])
    if 'combined' in request.modes:
        validate_training(store,{**request.candidate.model_dump(),'model_type':'combined',
            'reference_dataset_id':request.reference.dataset_id,'reference_split_id':request.reference.split_id,
            'class_mapping':request.reference_class_mapping})
    test = TestRequest(model_ids=['pending'], dataset_id=request.test_dataset_id, split_id=request.test_split_id,
                       class_mapping=request.test_class_mapping, acknowledge_missing_groups=request.acknowledge_missing_groups)
    test_source, names = test_manifest(store, test.model_dump())
    class_mapping(classes_for(store, request.test_dataset_id), canonical, request.test_class_mapping)
    assert_independent(manifests, test_source, names, request.acknowledge_missing_groups)
    if 'controlled' in request.modes and not store.get(request.candidate.dataset_id)['payload'].get('provenance', {}).get('artificial'):
        raise ValueError('Controlled mode requires a recorded perturbed candidate dataset')
    return {'test_sha256': test_source['test_checksum'], 'test_images': len(names),
            'dataset_versions': {c.dataset_id: store.get(c.dataset_id)['payload']['sha256'] for c in configurations},
            'requested_training_sizes': sizes,
            'training_budget': {k: getattr(request.candidate, k) for k in ('architecture', 'initialization', 'epochs', 'batch', 'image_size', 'learning_rate', 'device')},
            'control_notes': ['Reference/combined runs use the candidate architecture, initialization and training budget.',
                'Validation partitions are fixed per dataset; they may differ across datasets. Final independent test conditions are shared.',
                'Matched-size selection is deterministic, may exclude whole recording groups, and is reassessed before training.']}


def _matched_split(store, config, count, selection_seed):
    manifest = copy.deepcopy(get_split(store, config.split_id, config.dataset_id))
    # Keep complete known recording groups, avoiding arbitrary frame-level pruning.
    groups = {}
    for name in manifest['images']['train']:
        group = manifest.get('groups', {}).get(name) or name
        groups.setdefault(group, []).append(name)
    ordered = sorted(groups, key=lambda key: fingerprint([selection_seed, key]))
    selected = []
    for key in ordered:
        if len(selected) + len(groups[key]) <= count:
            selected.extend(groups[key])
    if len(selected) != count:
        raise ValueError('Exact matched size cannot be formed from complete recording groups; choose compatible datasets or natural sizes')
    manifest['images']['train'] = sorted(selected)
    manifest['parent_split_id'] = config.split_id
    manifest['selection'] = {'method': 'deterministic whole-group selection', 'seed': selection_seed, 'training_images': count}
    manifest.pop('checksum', None)
    manifest['checksum'] = fingerprint(manifest)
    return store.create('split', manifest)['id']


def execute_experiment(store, parent_job_id, config):
    from backend.workbench.worker import run
    experiment_id = config['experiment_id']
    record = store.get(experiment_id, 'experiment')['payload']
    request = ExperimentRequest.model_validate(record['request'])
    validate_experiment(store, request)
    children, models, evaluation_ids = [], [], []
    store.update(experiment_id, status='running')
    def execute(task, configuration):
        child = store.create('job', {'task': task, 'status': 'running', 'stage': 'Starting experiment step',
            'config': {**configuration, 'experiment_id': experiment_id}, 'parent_job_id': parent_job_id, 'error': None, 'result': None})
        children.append(child['id'])
        store.update(experiment_id, child_job_ids=children)
        store.update(parent_job_id, stage=f'{task}: step {len(children)}', progress={'completed_steps': len(children)-1})
        run(store, child['id'])
        finished = store.get(child['id'])['payload']
        if finished['status'] != 'completed':
            raise ValueError(f'Experiment {task} step failed: ' + str(finished.get('error')))
        return finished['result']
    candidate, reference = request.candidate.model_copy(), request.reference.model_copy() if request.reference else None
    if request.size_mode == 'matched' and reference:
        count = min(len(get_split(store, c.split_id, c.dataset_id)['images']['train']) for c in (candidate, reference))
        for c in (candidate, reference):
            c.split_id = _matched_split(store, c, count, request.selection_seed)
            old = store.get(c.readiness_id, 'job')['payload']['result']
            context = DatasetContext.model_validate(old['confirmed_context'])
            execute('readiness', {'dataset_id': c.dataset_id, 'split_id': c.split_id, 'context': context.model_dump()})
            c.readiness_id = children[-1]
            c.aggregation_id = None
    for mode in request.modes:
        source = reference if mode == 'reference' else candidate
        for seed in request.seeds:
            training = request.candidate.model_dump()
            training.update(dataset_id=source.dataset_id, split_id=source.split_id, readiness_id=source.readiness_id,
                aggregation_id=source.aggregation_id,
                validation_dataset_id=source.validation_dataset_id,validation_split_id=source.validation_split_id,
                validation_class_mapping=source.validation_class_mapping,acknowledge_validation_groups=source.acknowledge_validation_groups,
                model_type=mode, seed=seed, experiment_group=request.name,
                class_mapping=request.reference_class_mapping if mode == 'reference' else request.candidate.class_mapping)
            if mode == 'combined':
                training.update(reference_dataset_id=reference.dataset_id, reference_split_id=reference.split_id,
                                class_mapping=request.reference_class_mapping)
            result = execute('training', training)
            models.append(result['model_id'])
            store.update(experiment_id, model_ids=models)
    # The final test is invoked only after all candidate/reference training has ended.
    for start in range(0, len(models), 10):
        evaluation = TestRequest(model_ids=models[start:start+10], dataset_id=request.test_dataset_id, split_id=request.test_split_id,
            class_mapping=request.test_class_mapping, confidence=request.confidence, iou=request.iou, match_iou=request.match_iou,
            device=request.candidate.device, acknowledge_missing_groups=request.acknowledge_missing_groups)
        evaluation_ids.extend(execute('evaluation', evaluation.model_dump())['evaluation_ids'])
    store.update(experiment_id, status='completed', evaluation_ids=evaluation_ids)
    return {'experiment_id': experiment_id, 'model_ids': models, 'evaluation_ids': evaluation_ids}
