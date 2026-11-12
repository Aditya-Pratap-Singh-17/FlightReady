"""Subprocess entry point: HTTP requests never execute model training."""
import sys
import traceback
from backend.context import DatasetContext
from backend.readiness_engine import assess
from sql.store import Store
from backend.workbench.datasets import load, subset
from backend.workbench.splits import get_split


def run(store, id):
    job = store.get(id, 'job')['payload']
    config = job['config']
    try:
        if job['task'] == 'readiness':
            ds = load(store, config['dataset_id'])
            if config.get('split_id'):
                manifest = get_split(store, config['split_id'], config['dataset_id'])
                ds = subset(ds, manifest['images']['train'])
            def progress(stage):
                store.update(id, stage=stage)
                print(stage, flush=True)
            result = assess(ds, DatasetContext.model_validate(config['context']), progress=progress).model_dump(mode='json')
        elif job['task'] == 'training':
            from backend.workbench.detector import train
            result = train(store, id, config)
        elif job['task'] == 'evaluation':
            from backend.workbench.detector import evaluate_models
            result = evaluate_models(store, id, config)
        elif job['task'] == 'experiment':
            from backend.workbench.experiments import execute_experiment
            result = execute_experiment(store, id, config)
        elif job['task'] == 'model_acquisition':
            from backend.workbench.model_registry import acquire
            result = acquire(store, id, config)
        elif job['task'] == 'inference_benchmark':
            from backend.workbench.inference import benchmark
            result = benchmark(store, id, config)
        else:
            raise ValueError('Unknown job type')
        if store.get(id)['payload']['status'] != 'cancelled':
            store.update(id, status='completed', stage='Completed', result=result)
    except Exception as exc:
        traceback.print_exc()
        error = str(exc)
        if 'out of memory' in error.lower():
            error = 'CUDA out of memory. Reduce batch size (try 1 or 2) or image size (try 320), then create a new run. ' + error
        if store.get(id)['payload']['status'] != 'cancelled':
            store.update(id, status='failed', error=error, stage='Failed')
            if job['task'] == 'experiment':
                store.update(config['experiment_id'], status='failed', error=error)


if __name__ == '__main__':
    run(Store(), sys.argv[1])
