"""Readable PDF/HTML research exports from persisted evidence only."""
from html import escape
import copy
import io
import json
import math


def text(value, limit=None):
    value = json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, (dict, list)) else str(value if value is not None else 'Not assessed')
    return value if not limit or len(value) <= limit else value[:limit] + ' [abbreviated; see JSON export]'


def quality_html(profile):
    if not profile:
        return '<h2>Adaptive quality profile</h2><p>Not assessed in this historical report.</p>'
    bars, rows = [], []
    for i, d in enumerate(profile['dimensions']):
        y = 24 + i * 36
        bars.append(f'<text x="0" y="{y+13}" font-size="12">{escape(d["name"])}</text>')
        if d['score'] is not None:
            bars.append(f'<rect x="255" y="{y}" width="{d["score"]*3.2}" height="20" fill="#398870"/><text x="590" y="{y+14}" font-size="12">{d["score"]:.1f}</text>')
        else:
            bars.append(f'<text x="255" y="{y+14}" font-size="12">Insufficient evidence</text>')
        rows.append('<tr>'+''.join('<td>'+escape(text(v))+'</td>' for v in (d['name'], round(d['score'], 2) if d['score'] is not None else None,
            d['status'], f'{100*d["effective_weight"]:.1f}%', f'{100*d["evidence_coverage"]:.1f}%'))+'</tr>')
    return '<h2>Adaptive multi-dimensional readiness</h2><p>'+escape(profile['readiness_status'])+'</p><svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Measured dimension scores on a 0 to 100 scale" viewBox="0 0 650 270">'+''.join(bars)+'</svg><table><tr><th>Dimension</th><th>Score / 100</th><th>Evidence status</th><th>Importance</th><th>Coverage</th></tr>'+''.join(rows)+'</table><p>'+escape(profile['formula'])+'</p><p>'+escape(' '.join(profile['weight_explanations']))+'</p><details><summary>Critical gates, metric normalization, exact weights and provenance</summary><pre>'+escape(json.dumps(profile, indent=2))+'</pre></details>'


def research_evidence_html(certificate):
    evidence = certificate.get('quality_measurements', {}).get('research_evidence', {})
    if not evidence:
        return '<h2>Enhanced research evidence</h2><p>Unavailable in this historical certificate; reassess to collect new measurements.</p>'
    output = ['<h2>Enhanced research evidence</h2><p>'+escape(evidence['notice'])+'</p>']
    for section in evidence['sections']:
        output.append('<h3>'+escape(section['title'])+'</h3><p>'+escape(text({k:section[k] for k in ('metric_id','status','evidence_type','sample_size','coverage','denominator','version')}))+'</p>')
        for plot in section['plots']:
            output.append('<h4>'+escape(plot['title'])+'</h4><p>'+escape(text({'units':plot['unit'],'denominator':plot['denominator'],'status':plot['status']}))+'</p>')
            if plot['rows']:
                output.append('<table><tr><th>Category / interval</th><th>Measured value</th></tr>'+''.join('<tr><td>'+escape(str(row['label']))+'</td><td>'+escape(text(row['value']))+'</td></tr>' for row in plot['rows'])+'</table>')
            else:
                output.append('<p>Unavailable; no value inferred.</p>')
        output.append('<table><tr><th>Quantity</th><th>Value</th><th>Unit</th><th>Denominator</th></tr>'+''.join('<tr>'+''.join('<td>'+escape(text(row[k]))+'</td>' for k in ('label','value','unit','denominator'))+'</tr>' for row in section['rows'])+'</table>')
        output.append('<p>'+escape(' '.join(section['limitations']))+'</p><details><summary>Method and provenance</summary><pre>'+escape(text({'source':section['provenance'],'methodology':section['methodology']}))+'</pre></details>')
    return ''.join(output)


def document(store, record):
    payload = record['payload']
    if record['kind'] == 'benchmark':
        preview=payload.get('kind')=='inference_preview'
        return {'title':'Inference-only Benchmark Report', 'id':record['id'], 'created_at':record['created_at'],
            'status':'Completed real prediction preview' if preview else 'Completed real inference',
            'summary':'Actual model predictions; detection quality was not scored.' if preview else 'Measured object detection for the explicitly selected annotation classes. Review the declared scope; this need not validate the project goal.',
            'rows':[], 'sections': [('Observed performance',payload['metrics']), ('Model, revision and safe artifact hashes',payload['model']),
                ('Dataset and evaluation protocol',{k:payload[k] for k in ('readiness_id','dataset_id','dataset_sha256','test_sha256','configuration','test_images')}),
                ('Evaluation scope',payload.get('evaluation_scope',{})),
                ('Reproducibility',payload['reproducibility']),('Limitations',payload['limitations'])]}
    if record['kind']=='aggregation':
        source=copy.deepcopy(store.get(payload['request']['readiness_id'],'job'))
        source['id']=record['id']
        source['created_at']=record['created_at']
        source['payload']['result']['quality_profile']=payload['result']
        # A saved policy preview is a distinct artifact. Do not reuse the frozen
        # canonical verdict/hash from its source assessment for a different policy.
        source['payload']['result']['final_report']=None
        doc=document(store,source)
        doc['title']='Dataset Readiness Report - Saved Aggregation'
        doc['sections'].insert(0,('Saved aggregation audit trail',{'request':payload['request'],'source_assessment_id':payload['request']['readiness_id']}))
        if payload['result']['requires_measurement']:
            doc['status']='Incomplete task preview - new task-specific evidence required'
        return doc
    readiness = record['kind'] == 'job' and payload.get('task') == 'readiness' and payload.get('status') == 'completed'
    if readiness:
        cert = payload['result']
        final = cert.get('final_report')
        return {'title': 'Dataset Readiness Report', 'id': record['id'], 'created_at': record['created_at'],
            'status': final['decision'] if final else 'Confirmed requirements: '+cert['overall_status']+' | Adaptive evidence profile: '+(cert.get('quality_profile') or {}).get('readiness_status','not assessed'),
            'summary': final['conclusion'] if final else cert['executive_summary'], 'certificate': cert, 'final_report': final,
            'quality_profile': cert.get('quality_profile'), 'sections': [
                ('Dataset identity and provenance', cert['dataset']), ('Confirmed task and original context', cert['confirmed_context']),
                ('Methodology and versions', cert.get('quality_profile', {})),
                ('Critical requirements', cert['findings']), ('Recommendations', cert['recommendations']),
                ('Measurement limitations', cert['limitations']), ('Reproducibility', cert['reproducibility'])], 'rows': []}
    from backend.workbench.analytics import research
    from backend.workbench.conclusions import experiment_conclusion
    data = research(store)
    if record['kind'] == 'experiment':
        ids = set(payload.get('evaluation_ids', []))
        # Child evaluations may exist even when a later experiment step failed.
        child_ids = set(payload.get('child_job_ids', []))
        for evaluation in store.list('evaluation'):
            if evaluation['payload'].get('job_id') in child_ids:
                ids.add(evaluation['id'])
        rows = [r for r in data['rows'] if r['evaluation_id'] in ids]
        complete = payload.get('status') == 'completed'
    elif record['kind'] == 'evaluation':
        rows = [r for r in data['rows'] if r['evaluation_id'] == record['id']]
        complete = True
    elif record['kind'] == 'comparison':
        rows = [r for r in data['rows'] if r['evaluation_id'] in payload['evaluation_ids']]
        complete = bool(rows)
    elif record['kind'] == 'job':
        if payload.get('task')=='training':
            model_id=(payload.get('result') or {}).get('model_id')
            rows=[r for r in data['rows'] if r['model_id']==model_id]
        elif payload.get('task')=='evaluation':
            ids=(payload.get('result') or {}).get('evaluation_ids',[])
            rows=[r for r in data['rows'] if r['evaluation_id'] in ids]
        elif payload.get('task')=='experiment':
            return document(store,store.get(payload['config']['experiment_id'],'experiment'))
        else:
            rows=[]
        complete=payload.get('status')=='completed' and bool(rows)
    else:
        rows = data['rows']
        complete = bool(rows)
    models = [store.get(r['model_id'], 'model') for r in rows]
    readiness_records = [store.get(m['payload']['readiness_id'], 'job') for m in models]
    scoped_analysis=research(store,model_ids={r['model_id'] for r in rows})
    return {'title': 'Model Validation Research Report', 'id': record['id'], 'created_at': record['created_at'],
        'status': 'Completed' if complete else 'Incomplete - missing experimental outcomes',
        'summary': 'Recorded training and independent-test results. Repeated seeds are not independent datasets. Correlations are exploratory; no universal predictive validity is established.',
        'rows': rows, 'quality_profiles': [{'model_id':m['id'],'profile':m['payload'].get('quality_profile') or r['payload']['result'].get('quality_profile'),
            'scope':m['payload'].get('quality_evidence_scope','Associated training-subset assessment; inspect source manifests')} for m,r in zip(models,readiness_records)],
        'sections': ([('Scientific validation conclusion', experiment_conclusion(store,record))] if record['kind']=='experiment' else []) + [('Experiment objective, hypothesis and protocol', payload),
            ('Training configurations and model provenance', models), ('Training-data readiness evidence', readiness_records),
            ('Quality-performance analysis for the selected models', {k: scoped_analysis.get(k) for k in ('correlations', 'predictive_validity', 'intervention_pairs', 'summary')}),
            ('Experimental limitations', ['Final tests must remain excluded from model selection and hyperparameter tuning.',
                'Different source distributions, validation partitions, dataset sizes and compute conditions can confound comparisons.',
                'Missing model results are not estimated. See JSON for complete raw evidence and split manifests.'])]}


def html_document(doc):
    if doc.get('final_report'):
        return final_html(doc)
    rows = doc['rows']
    table = ''
    if rows:
        headings = ['Dataset / version', 'Architecture / seed', 'Test dataset', 'mAP50', 'mAP50:95', 'Precision', 'Recall', 'Latency ms']
        values = [[r['dataset']+' / '+r['dataset_version'], str(r['configuration'].get('architecture'))+' / '+str(r['seed']), r['test_dataset'],
            *[r['metrics'].get(k) for k in ('mAP50', 'mAP50_95', 'precision', 'recall', 'inference_latency_ms')]] for r in rows]
        table = '<h2>Measured model outcomes</h2><table><tr>'+''.join('<th>'+h+'</th>' for h in headings)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(text(v))+'</td>' for v in row)+'</tr>' for row in values)+'</table>'
    profiles=''.join('<h2>Model '+escape(q['model_id'])+'</h2><p>'+escape(q['scope'])+'</p>'+quality_html(q['profile']) for q in doc.get('quality_profiles',[]))
    profiles+=research_evidence_html(doc['certificate']) if doc.get('certificate') else ''
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+escape(doc['title'])+'</title><style>body{font:15px system-ui;color:#173a32;max-width:1050px;margin:40px auto;padding:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dce4df;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}svg{max-width:850px;width:100%}h2{margin-top:32px}</style><h1>'+escape(doc['title'])+'</h1><p>'+escape(doc['status'])+'</p><p>'+escape(doc['summary'])+'</p><p>Record '+escape(doc['id'])+' · '+escape(doc['created_at'])+'</p>'+quality_html(doc.get('quality_profile'))+table+profiles+''.join('<h2>'+escape(title)+'</h2><pre>'+escape(json.dumps(value,indent=2,ensure_ascii=True))+'</pre>' for title,value in doc['sections'])+'</html>'


def pdf_document(doc):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallEvidence', fontName='Helvetica', fontSize=8, leading=11, spaceAfter=5, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='Executive', fontName='Helvetica', fontSize=8.5, leading=11, spaceAfter=4, wordWrap='CJK'))
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    story = []
    def p(value, style='BodyText'):
        # Standard PDF fonts use ASCII for portable, predictable glyph rendering.
        safe = text(value).encode('ascii', 'replace').decode()
        return Paragraph(escape(safe).replace('\n', '<br/>'), styles[style])
    def heading(title): story.extend([Spacer(1, 10), p(title, 'Heading2')])
    def table(headings, values, widths, compact=False):
        rows = [[p(v, 'SmallEvidence') for v in row] for row in [headings, *values]]
        result = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
        result.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9f1ed')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 3 if compact else 8),
            ('LINEBELOW', (0, 0), (-1, -1), .35, colors.HexColor('#d8e3dc'))]))
        story.append(result)
    story.extend([p('FLIGHTREADY / RESEARCH EVIDENCE', 'SmallEvidence'), p(doc['title'], 'Title'), p(doc['status'], 'Heading2'),
                  p(doc['summary']), Spacer(1, 8), p('Record: '+doc['id']+' | '+doc['created_at'], 'SmallEvidence')])
    final = doc.get('final_report')
    if final:
        def executive_heading(value):
            story.extend([Spacer(1,5),p(value,'Heading3')])
        executive_heading('A. Final verdict')
        story.extend(p(v,'Executive') for v in [final['dataset']['name']+' | Intended task: '+final['intended_task']['task_description'],
            'Dataset SHA256: '+final['dataset']['sha256'], 'Task compatibility: '+final['compatibility']['status']+'. '+final['compatibility']['summary'],
            'Evidence coverage: '+(f'{100*final["evidence_coverage"]:.1f}%' if final['evidence_coverage'] is not None else 'not assessed')+' | Critical blockers: '+str(final['critical_blockers']),
            'Overall index: not scored. '+final['criterion_attainment_index']['reason']])
        executive_heading('B. Six quality dimensions')
        table(['Dimension', 'Score / 100', 'Coverage', 'Importance'], [[d['name'],d['score'],f'{d["evidence_coverage"]*100:.1f}%',f'{d["effective_weight"]*100:.1f}%'] for d in final['dimensions']], [215,80,100,100], compact=True)
        executive_heading('C. What compromises readiness?')
        for item in final['top_problems']:
            story.append(p(item['priority']+': '+item['reason'],'Executive'))
        if not final['top_problems']:
            story.append(p('No confirmed deficiencies in the configured criteria. Review the limitations.'))
        executive_heading('D. Recommended improvements')
        for item in final['recommendations'][:3]:
            story.append(p(item['action'],'Executive'))
        executive_heading('E. Validation status')
        story.append(p(final['validation']['status']+': '+final['validation']['reason'],'Executive'))
        story.append(p('Assessment '+final['assessment_fingerprint']+' | Report '+final['report_version']+' | '+final['report_sha256'], 'SmallEvidence'))
        story.append(PageBreak())
        heading('F. Technical appendix')
    profile = doc.get('quality_profile')
    if profile:
        heading('Multi-dimensional quality profile')
        drawing = Drawing(495, 210)
        for i, d in enumerate(profile['dimensions']):
            y = 180 - i*28
            drawing.add(String(0, y+3, d['name'], fontName='Helvetica', fontSize=8))
            drawing.add(Rect(210, y, 240, 16, fillColor=colors.HexColor('#edf2ef'), strokeColor=None))
            if d['score'] is not None:
                drawing.add(Rect(210, y, 2.4*d['score'], 16, fillColor=colors.HexColor('#398870'), strokeColor=None))
                drawing.add(String(458, y+3, f'{d["score"]:.1f}', fontSize=8))
            else:
                drawing.add(String(215, y+3, 'Not scored: incomplete evidence', fontSize=8))
        drawing.add(String(210, 0, '0                       Measured criterion quality                         100', fontSize=7))
        story.append(drawing)
        table(['Dimension', 'Score / 100', 'Status', 'Importance', 'Coverage'], [[d['name'], round(d['score'], 2) if d['score'] is not None else None,
            d['status'], f'{d["effective_weight"]*100:.1f}%', f'{d["evidence_coverage"]*100:.1f}%'] for d in profile['dimensions']], [155, 65, 115, 80, 80])
        heading('Scoring methodology and provenance')
        story.extend(p(v) for v in [profile['formula'], *profile['weight_explanations'], *profile['limitations']])
        story.append(p('Versions: '+profile['profile_version']+' / '+profile['normalization_version']+' / '+profile['aggregation_version']))
        heading('Critical requirements (independent of weights)')
        table(['Requirement', 'Severity / state', 'Measured finding'], [[g['id'], g['severity']+' / '+g['status'], g['reason']] for g in profile['critical_requirements']], [110, 110, 275])
        heading('Component normalization and exact within-dimension weights')
        table(['Metric / direction', 'Normalized score', 'Weight / state', 'Evidence / limitations'], [[m['label']+' / '+m['definition']['direction'], m['score'],
            f'{m["effective_weight"]:.6f} / {m["status"]}', m['normalization']+'; '+m['reason']] for m in profile['individual_metrics']], [130, 65, 100, 200])
    for entry in doc.get('quality_profiles',[]):
        training_profile=entry['profile']
        heading('Training readiness profile: model '+entry['model_id'][:12])
        story.append(p(entry['scope']))
        if training_profile:
            table(['Dimension','Score / 100','Evidence coverage','Importance'],[[d['name'],d['score'],f'{100*d["evidence_coverage"]:.1f}%',f'{100*d["effective_weight"]:.1f}%'] for d in training_profile['dimensions']],[190,95,110,100])
            story.append(p('Scoring policy '+training_profile['profile_version']+'; scenario '+training_profile['scenario']+'; calculation SHA256 '+training_profile['calculation_sha256'],'SmallEvidence'))
    if doc.get('certificate'):
        cert = doc['certificate']
        heading('Enhanced research measurements')
        evidence=cert.get('quality_measurements',{}).get('research_evidence',{})
        story.append(p(evidence.get('notice','Unavailable in this historical certificate; run a new assessment.')))
        for section in evidence.get('sections',[]):
            heading(section['title'])
            story.append(p({k:section[k] for k in ('metric_id','status','evidence_type','sample_size','coverage','denominator','version')},'SmallEvidence'))
            for plot in section['plots']:
                story.append(p(plot['title']+' | '+plot['unit']+' | denominator '+text(plot['denominator'])+' | '+plot['status'],'SmallEvidence'))
                if plot['rows']:
                    table(['Category / interval','Measured value'],[[r['label'],r['value']] for r in plot['rows']],[345,150])
                else:
                    story.append(p('Unavailable; no value inferred.'))
            if section['rows']:
                table(['Quantity','Value','Unit','Denominator'],[[r[k] for k in ('label','value','unit','denominator')] for r in section['rows']],[190,65,100,140])
            story.append(p(' '.join(section['limitations']),'SmallEvidence'))
            story.append(p(text({'provenance':section['provenance'],'methodology':section['methodology']},3500),'SmallEvidence'))
        heading('All raw metrics')
        story.append(p('Long structured values are abbreviated in this readable report. The JSON export retains every value and supporting reference.'))
        table(['Metric', 'Measurement state', 'Raw value (abbreviated)'], [[m['metric_id'], m.get('assessment_status') or m['state'], text(m['value'], 160)] for m in cert['metrics']], [105, 100, 290])
    if doc['rows']:
        heading('Independent test outcomes')
        table(['Dataset / seed', 'Model', 'Test', 'mAP 50:95', 'Precision / recall'], [[r['dataset']+' / '+str(r['seed']), r['model_id'][:10], r['test_dataset'],
            r['metrics'].get('mAP50_95'), str(r['metrics'].get('precision'))+' / '+str(r['metrics'].get('recall'))] for r in doc['rows']], [125, 85, 125, 65, 95])
        drawing = Drawing(495, min(480, 40+28*len(doc['rows'])))
        for i, r in enumerate(doc['rows'][:15]):
            y = drawing.height-30-i*28
            drawing.add(String(0, y+4, r['model_id'][:12]+' / seed '+str(r['seed']), fontSize=8))
            value = r['metrics'].get('mAP50_95')
            if value is not None:
                drawing.add(Rect(175, y, max(0, value)*260, 16, fillColor=colors.HexColor('#398870'), strokeColor=None))
                drawing.add(String(440, y+4, f'{value:.4f}', fontSize=8))
        drawing.add(String(175, 0, 'mAP@0.50:0.95 (0-1); descriptive across unmatched protocols', fontSize=7))
        story.append(drawing)
    # Retain meaningful context and reproducibility, with explicit abbreviation.
    for title, value in doc['sections']:
        if title in ('Methodology and versions', 'Critical requirements') and profile:
            continue
        heading(title)
        if isinstance(value, list):
            for entry in value:
                story.append(p(text(entry, 2500), 'SmallEvidence'))
        else:
            story.append(p(text(value, 7000), 'SmallEvidence'))
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#60766b'))
        canvas.drawString(40, 25, 'FlightReady | Recorded evidence; no fabricated outcomes')
        canvas.drawRightString(A4[0]-40, 25, str(document.page))
    SimpleDocTemplate(output, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=45,
                      title=doc['title'], author='FlightReady Research Lab').build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def final_html(doc):
    """One-minute verdict; full evidence remains available through disclosure."""
    final = doc['final_report']
    def e(value): return escape(text(value))
    def issues(items):
        return ''.join('<li><strong>'+e(i['priority'])+'</strong>: '+e(i['reason'])+'<p>'+e(i['action'])+'</p></li>' for i in items) or '<li>No confirmed deficiencies in the configured criteria.</li>'
    rows = ''.join('<tr>'+''.join('<td>'+e(v)+'</td>' for v in (d['name'], d['score'] if d['score'] is not None else 'Not scored',
        f'{100*d["evidence_coverage"]:.1f}%', f'{100*d["effective_weight"]:.1f}%', d['status']))+'</tr>' for d in final['dimensions'])
    coverage = f'{100*final["evidence_coverage"]:.1f}%' if final['evidence_coverage'] is not None else 'Not assessed'
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlightReady final readiness report</title>'
        '<style>body{font:16px/1.6 system-ui;color:#173a32;max-width:1050px;margin:32px auto;padding:24px}h1{line-height:1.2}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd9d2;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{margin:18px 0}summary{cursor:pointer;font-weight:650}small{overflow-wrap:anywhere}section{margin:28px 0}@media print{details{display:block}}</style>'
        '<h1>FlightReady final readiness report</h1><section><h2>A. Final verdict: '+e(final['decision'])+'</h2><p>'+e(final['conclusion'])+'</p>'
        '<p><strong>'+e(final['dataset']['name'])+'</strong><br>Intended task: '+e(final['intended_task']['task_description'])+'</p><p>Task compatibility: '+e(final['compatibility']['status'])+'</p><p>'+e(final['compatibility']['summary'])+'</p>'
        '<p>Evidence coverage: '+e(coverage)+' | Critical blockers: '+e(final['critical_blockers'])+'</p><p>Overall index: Not scored. '+e(final['criterion_attainment_index']['reason'])+'</p><small>Dataset SHA256: '+e(final['dataset']['sha256'])+'<br>Assessment fingerprint: '+e(final['assessment_fingerprint'])+'<br>Report version: '+e(final['report_version'])+'<br>Report SHA256: '+e(final['report_sha256'])+'</small></section>'
        '<section><h2>B. Six quality dimensions</h2><table><tr><th>Dimension</th><th>Score / 100</th><th>Coverage</th><th>Importance</th><th>Status</th></tr>'+rows+'</table><p>'+e(final['coverage_definition'])+'</p></section>'
        '<section><h2>C. What compromises readiness?</h2><ol>'+issues(final['top_problems'])+'</ol><details><summary>Hard requirements and evidence gaps</summary><pre>'+e(final['hard_requirements'])+'</pre></details></section>'
        '<section><h2>D. Recommended improvements</h2><ol>'+issues(final['recommendations'][:3])+'</ol><details><summary>All recommendations</summary><ol>'+issues(final['recommendations'])+'</ol></details></section>'
        '<section><h2>E. Validation status</h2><p>'+e(final['validation']['status'])+': '+e(final['validation']['reason'])+'</p><p>Model outcomes are a separately linked supplement. They do not rewrite this assessment.</p></section>'
        '<section><h2>F. Technical appendix</h2><details><summary>Methods, bounds, provenance and limitations</summary><pre>'+e(final)+'</pre></details><details><summary>All original measurements and reproducibility</summary>'+research_evidence_html(doc['certificate'])+'<pre>'+e(doc['certificate'])+'</pre></details></section></html>')
