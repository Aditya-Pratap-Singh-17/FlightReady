"""Scientific interpretations assembled exclusively from persisted experiments."""
from backend.workbench.analytics import comparison


def experiment_conclusion(store, record):
    payload=record['payload']; request=payload['request']
    ids=payload.get('evaluation_ids',[])
    evaluations=[store.get(id,'evaluation') for id in ids]
    reference=request.get('reference')
    if not reference or not evaluations:
        return {'status':'insufficient_comparison_evidence','pairs':[],
                'conclusion':'A baseline and candidate need completed evaluation on a common held-out protocol. No performance difference is inferred.'}
    candidate=request['candidate']['dataset_id']; baseline=reference['dataset_id']
    # In the guided degradation study, the unmodified source is the baseline.
    if store.get(baseline,'dataset')['payload'].get('provenance',{}).get('artificial') and not store.get(candidate,'dataset')['payload'].get('provenance',{}).get('artificial'):
        baseline,candidate=candidate,baseline
    entries=[(e,store.get(e['payload']['model_id'],'model')['payload']) for e in evaluations]
    pairs=[]
    for before,bmodel in entries:
        if bmodel['dataset_id']!=baseline: continue
        for after,amodel in entries:
            if amodel['dataset_id']!=candidate or bmodel['configuration']['seed']!=amodel['configuration']['seed']: continue
            a,b=before['payload'],after['payload']
            if a['test_sha256']!=b['test_sha256'] or a['settings']!=b['settings']: continue
            audit=comparison(store,[before['id'],after['id']])
            deltas={}
            for metric in ('mAP50','mAP50_95','precision','recall'):
                av,bv=a['metrics'].get(metric),b['metrics'].get(metric)
                if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
                    deltas[metric]={'baseline':av,'candidate':bv,'absolute_change':bv-av,'relative_change':(bv-av)/av if av else None,
                        'unit':'fraction; multiply absolute changes by 100 for percentage points'}
            bc=store.get(bmodel['readiness_id'],'job')['payload']['result']
            ac=store.get(amodel['readiness_id'],'job')['payload']['result']
            bp=bmodel.get('quality_profile') or bc.get('quality_profile') or {}
            ap=amodel.get('quality_profile') or ac.get('quality_profile') or {}
            def policy(profile):
                return [profile.get(k) for k in ('profile_sha256','task','scenario','normalization_version','aggregation_version','confirmed_overrides')]+[
                    [(d['id'],d['effective_weight'],[(m['id'],m['effective_weight']) for m in d['metrics']]) for d in profile.get('dimensions',[])]]
            same_policy=bool(bp and ap) and policy(bp)==policy(ap)
            quality=[]
            for dim in bp.get('dimensions',[]):
                other=next((d for d in ap.get('dimensions',[]) if d['id']==dim['id']),None)
                if other:
                    x,y=dim['score'],other['score']
                    quality.append({'dimension':dim['id'],'baseline_score':x,'candidate_score':y,
                        'absolute_change':y-x if same_policy and x is not None and y is not None else None,
                        'baseline_coverage':dim['evidence_coverage'],'candidate_coverage':other['evidence_coverage']})
            change=deltas.get('mAP50_95')
            observed=(f"For seed {bmodel['configuration']['seed']}, mAP@0.50:0.95 changed from {change['baseline']:.6f} to {change['candidate']:.6f} "
                f"(absolute change {change['absolute_change']:+.6f}) on {a['test_images']} common held-out images." if change else 'No jointly measured mAP result is available.')
            pairs.append({'baseline_evaluation_id':before['id'],'candidate_evaluation_id':after['id'],'seed':bmodel['configuration']['seed'],
                'protocol_review':audit,'performance_changes':deltas,'quality_policy_matches':same_policy,'quality_changes':quality,
                'intervention':store.get(candidate,'dataset')['payload'].get('provenance',{}),
                'conclusion':observed+(' Recorded training conditions and sample counts match.' if audit['matched_conditions'] else ' Protocol or sample-count differences prevent a controlled interpretation.'),
                'limitations':['A same-seed paired observation is not an independent dataset or evidence of statistical significance.',
                    'Quality changes are measured under the declared policy; unknown scores have no numerical change.',
                    'These results do not validate a universal readiness-to-accuracy relationship or establish causation.']})
    return {'status':'observed_comparisons' if pairs else 'insufficient_comparison_evidence','pairs':pairs,
        'conclusion':' '.join(p['conclusion'] for p in pairs) if pairs else 'No completed same-seed baseline/candidate pair shares the same held-out evaluation protocol.',
        'next_steps':'Use repeated seeds, matched sample-count controls where needed, independent datasets and a prespecified hypothesis before making broader claims.'}
