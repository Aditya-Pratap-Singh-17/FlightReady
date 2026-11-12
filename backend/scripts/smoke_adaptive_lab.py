"""Actual named experiment/perturbation smoke against an isolated local fixture server.

Run scripts/smoke_workbench.py first, then serve its storage with WORKBENCH_ROOT.
No external providers or real user datasets are accessed by this script.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import httpx

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from backend.workbench.contracts import TrainRequest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8049')
    parser.add_argument('--output',default='.runtime/adaptive-lab-smoke')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with httpx.Client(base_url=args.url+'/api/v1',timeout=120) as client:
        def get(path):
            response=client.get(path);response.raise_for_status();return response.json()
        def post(path,body=None):
            response=client.post(path,json=body);response.raise_for_status();return response.json()
        def wait(id):
            deadline=time.monotonic()+600;last=None
            while time.monotonic()<deadline:
                job=get('/jobs/'+id)['payload']
                stage=job.get('stage')
                if stage!=last:
                    print(stage,flush=True);last=stage
                if job['status']=='completed':return job['result']
                if job['status'] not in ('queued','running'):raise RuntimeError(job.get('error') or job['status'])
                time.sleep(.4)
            raise TimeoutError('Named experiment exceeded ten minutes')
        models=get('/models')
        sources=[]
        for mode in ('candidate','reference'):
            model=next(m for m in models if m['payload']['configuration']['model_type']==mode)
            config={k:v for k,v in model['payload']['configuration'].items() if k in TrainRequest.model_fields}
            config.update(epochs=1,batch=2,image_size=64,device='cpu')
            sources.append(config)
        candidate,reference=sources
        for configuration in sources:
            configuration.update(validation_dataset_id=reference['dataset_id'],validation_split_id=reference['split_id'])
        external=get('/evaluations')[0]['payload']['dataset_id']
        saved=post('/quality/aggregate',{'readiness_id':candidate['readiness_id'],'scenario':'airport_drone_detection',
            'metric_weights':{'clarity':1.25},'overrides_confirmed':True,'persist':True,'use_current_profile':True})
        candidate['aggregation_id']=saved['aggregation_id']
        protocol={'name':'Synthetic adaptive lab verification','hypothesis':'Software workflow smoke only; not scientific validation.',
            'candidate':candidate,'reference':reference,'modes':['candidate','reference','combined'],'seeds':[42],
            'test_dataset_id':external,'size_mode':'natural'}
        experiment=post('/experiments',protocol)
        result=wait(post('/experiments/'+experiment['id']+'/start')['id'])
        assert len(result['model_ids'])==3 and len(result['evaluation_ids'])==3
        evaluations={r['id']:r['payload'] for r in get('/evaluations')}
        assert all(evaluations[id]['test_sha256']==experiment['payload']['protocol']['test_sha256'] for id in result['evaluation_ids'])
        trained=get('/models')
        selected=next(m for m in trained if m['id']==result['model_ids'][0])
        assert selected['payload']['quality_profile']['calculation_sha256']==saved['calculation_sha256']
        combined=next(m for m in trained if m['id']==result['model_ids'][2])
        combined_report=get('/jobs/'+combined['payload']['readiness_id'])['payload']['result']
        assert combined_report['dataset']['profile']['image_count']==12
        assert combined['payload']['quality_evidence_scope'].startswith('Complete combined')
        changed=post('/variations',{'dataset_id':candidate['dataset_id'],'split_id':candidate['split_id'],
                                  'intervention':'blur','severity':.3,'seed':42})
        split=post('/datasets/'+changed['id']+'/splits',{'group_column':'flight_id'})
        source_readiness=get('/jobs/'+candidate['readiness_id'])['payload']
        ready=post('/readiness',{'dataset_id':changed['id'],'split_id':split['id'],'context':source_readiness['config']['context']})
        wait(ready['id'])
        changed_score=post('/quality/aggregate',{'readiness_id':ready['id'],'scenario':'airport_drone_detection',
            'metric_weights':{'clarity':1.25},'overrides_confirmed':True,'persist':True})
        perturbed={**candidate,'dataset_id':changed['id'],'split_id':split['id'],'readiness_id':ready['id'],
                   'aggregation_id':changed_score['aggregation_id'],'model_type':'controlled'}
        second=post('/experiments',{**protocol,'name':'Synthetic blur intervention verification','candidate':perturbed,'reference':None,'modes':['controlled']})
        wait(post('/experiments/'+second['id']+'/start')['id'])
        research=get('/research?metric=baseline.training_images')
        assert research['intervention_pairs']
        for kind,id in [('dataset',candidate['readiness_id']),('experiment',experiment['id'])]:
            for format in ('json','csv','html','pdf'):
                response=client.get(f'/reports/{id}?format={format}');response.raise_for_status()
                (out/f'{kind}.{format}').write_bytes(response.content)
        verification={'status':'passed','scope':'Real CPU compute on explicitly synthetic fixtures; no scientific predictive-validity claim.',
            'experiment_id':experiment['id'],'perturbation_experiment_id':second['id'],'model_ids':result['model_ids'],
            'evaluation_ids':result['evaluation_ids'],'intervention_pairs':len(research['intervention_pairs']),
            'scoring_version_link_verified':True}
        (out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
        print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
