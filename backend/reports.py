"""Deterministic reports: escaped evidence only, never generated metric claims."""
from html import escape
import json


def v2_score_html(scorecard):
    if not scorecard:
        return ''
    from metrics.readiness_v2_reporting import geometric_influence
    explanation = geometric_influence(scorecard)
    return ('<h2>FlightReady v2 score explanation</h2><p>V2 uses a weighted geometric mean over measured applicable metrics. '
            'The categorical readiness decision is determined separately by evidence coverage and gates.</p><p>'+
            escape(explanation['message'])+'</p><details><summary>Geometric aggregation diagnostics</summary><pre>'+
            escape(json.dumps(explanation, indent=2))+'</pre></details><details><summary>Frozen v2 scorecard</summary><pre>'+
            escape(json.dumps(scorecard, indent=2))+'</pre></details>')

def html_report(certificate):
    c = certificate
    score = c.experimental_score
    score_html = ''
    if score and score.get('status') != 'disabled':
        value = 'Unavailable' if score.get('score_percent') is None else f"{score['score_percent']:.2f} / 100"
        score_html = ('<h2>Experimental weighted quality score</h2><p>'+escape(value)+' ('+escape(score['status'])+
            ')</p><p>Evidence coverage: '+escape(str(round(score.get('evidence_coverage_percent',0),2)))+
            '% of applicable weight. '+escape(score['notice'])+'</p><details><summary>Weights, contributions, missing evidence and score profile</summary><pre>'+escape(json.dumps(score,indent=2))+'</pre></details>')
    from backend.workbench.reporting import quality_html
    score_html = quality_html(c.quality_profile) + score_html
    from backend.workbench.reporting import research_evidence_html
    score_html += research_evidence_html(c.model_dump(mode='json'))
    def section(title, items):
        return '<h2>'+escape(title)+'</h2><ul>'+''.join('<li>'+escape(str(x))+'</li>' for x in items)+'</ul>'
    findings = ''.join('<tr><td>'+escape(f.requirement.description)+'</td><td>'+f.requirement.severity+'</td><td>'+f.state+'</td><td>'+escape(f.reason)+'</td></tr>' for f in c.findings)
    metrics = ''.join('<details><summary>'+escape(m.dimension+' / '+m.metric_id+' — '+(m.assessment_status or m.state))+'</summary><pre>'+escape(m.model_dump_json(indent=2))+'</pre></details>' for m in c.metrics)
    views='<h2>Learning-potential diagnostics</h2><pre>'+escape(json.dumps(c.learning_views,indent=2))+'</pre>' if c.learning_views else ''
    metrics+=views+v2_score_html(c.quality_profile_v2)+score_html
    inventory={status:[m.metric_id for m in c.metrics if m.assessment_status==status] for status in
        ('measured','estimated','insufficient_evidence','not_applicable','unsupported','error')}
    metrics+='<h2>Measurement availability</h2><pre>'+escape(json.dumps(inventory,indent=2))+'</pre>'
    groups={
        'Environmental and scenario coverage':{'UAV-%02d'%i for i in range(1,8)},
        'Object-level coverage and observability':{'UAV-%02d'%i for i in range(8,15)},
        'Annotation diagnostics':{'UAV-15','UAV-16'},
        'Deployment, independent evidence and leakage':{'UAV-17','UAV-18'},
        'Sensor and temporal findings':{'UAV-19','UAV-20'},
    }
    metrics+=''.join(section(title,[m.metric_id+': '+(m.assessment_status or m.state) for m in c.metrics if m.metric_id in ids]) for title,ids in groups.items())
    metrics+='<h2>Recommendation evidence</h2><pre>'+escape(json.dumps([r.model_dump(mode='json') for r in c.recommendations],indent=2))+'</pre>'
    metrics+='<h2>Dataset provenance</h2><pre>'+escape(json.dumps(c.dataset.get('provenance',{}),indent=2))+'</pre>'
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Readiness report</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#183a38}td,th{text-align:left;border-bottom:1px solid #ddd;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{padding:10px;cursor:pointer}h1{font-size:32px}</style><h1>Dataset readiness assessment</h1><p>'+escape(c.disclaimer)+'</p><h2>'+escape(c.overall_status)+'</h2><p>'+escape(c.executive_summary)+'</p><p>Dataset: '+escape(c.dataset['name'])+'<br>SHA256: '+escape(c.dataset['sha256'])+'</p><h2>Your goal</h2><p>'+escape(c.confirmed_context.original_text)+'</p><details><summary>Initial and confirmed context</summary><pre>'+escape(json.dumps({'initial':c.initial_context.model_dump() if c.initial_context else None,'confirmed':c.confirmed_context.model_dump()},indent=2))+'</pre></details><h2>Requirement findings</h2><table><tr><th>Requirement</th><th>Severity</th><th>State</th><th>Evidence and reason</th></tr>'+findings+'</table><h2>Eight-dimensional assessment</h2>'+metrics+section('Additional benefits',c.additional_benefits)+section('Limitations and risks',c.limitations)+section('Remediation',[r.description for r in c.recommendations])+'<details><summary>Reproducibility</summary><pre>'+escape(json.dumps(c.reproducibility,indent=2))+'</pre></details></html>'
