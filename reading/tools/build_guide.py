"""Assemble the visual textbook from curated prose and verified source inventories."""
from pathlib import Path
import ast
import json
import re
from collections import Counter
from manuscript import FILE_ROLES, CHAPTERS, TEXT, GLOSSARY, DEBUG
from deep_reference import FUNCTIONS
from diagrams import DIAGRAMS
from current_pipeline import SNAPSHOT

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'reading'
ASSETS=DOC/'assets/architecture'
PY=json.loads((ASSETS/'source-inventory.json').read_text(encoding='utf-8'))
TS=json.loads((ASSETS/'typescript-inventory.json').read_text(encoding='utf-8'))
FILES={f['path']:f for f in PY['python']+TS}
SYMS={(f['path'],s['name']):s for f in FILES.values() for s in f['symbols']}

def link(path,name=None):
    if not (ROOT/path).is_file():raise ValueError('Missing source '+path)
    line=SYMS[(path,name)]['line'] if name else None
    label=(path+(' : '+name if name else '')).replace('_','\\_')
    return f'[{label}](../{path}{"#L"+str(line) if line else ""})'

def clean(value):
    return str(value).replace('|','\\|').replace('\n',' ').replace('\r',' ')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(clean(c) for c in row)+' |' for row in rows])+'\n'

def anchor(id):return f'<a id="{id}"></a>\n'

def diagram(id):
    d=next(d for d in DIAGRAMS if d['id']==id)
    (ASSETS/(id+'.mmd')).write_text(d['source']+'\n',encoding='utf-8')
    return '\n'+anchor('diagram-'+id)+f'### Figure: {d["title"]}\n\n<!-- diagram: {id} -->\n\n```mermaid\n{d["source"]}\n```\n\n{d["explanation"]}\n\n'+'Sources: '+', '.join(link(p) for p in d['files'])+f'. [Open SVG](assets/architecture/{id}.svg) | [High-resolution PNG](assets/architecture/{id}.png)\n'

CHAPTER_SOURCES={
'ch01':['main.py','backend/readiness_engine.py','metrics/quality_aggregation.py','metrics/experimental_scoring.py','frontend/src/GuidedWorkflow.tsx'],
'ch02':list(FILE_ROLES),
'ch03':['main.py','backend/workbench/api.py','sql/store.py','backend/workbench/jobs.py','frontend/src/main.tsx','frontend/src/Workbench.tsx','frontend/vite.config.ts','frontend/package.json'],
'ch04':['frontend/src/Workbench.tsx','frontend/src/GuidedWorkflow.tsx','frontend/src/GuidedValidation.tsx','frontend/src/api.ts','frontend/src/useAssessment.ts','frontend/src/QualityAnalytics.tsx','frontend/src/BenchmarkCharts.tsx'],
'ch05':['main.py','backend/workbench/api.py','backend/workbench/contracts.py','backend/workbench/workflow_api.py','backend/workbench/quality_api.py','backend/workbench/lab_api.py'],
'ch06':['backend/ingestion.py','backend/large_ingestion.py','backend/record_formats.py','backend/dataset_archives.py','backend/image_ingestion.py','backend/universal_images.py','backend/workbench/datasets.py','backend/workbench/combined.py'],
'ch07':['backend/context.py','backend/context_extractor.py','backend/workbench/workflow_api.py','backend/workbench/api.py'],
'ch08':['metrics/core.py','metrics/disk_metrics.py','backend/readiness_engine.py','metrics/uav_metrics/registry.py','metrics/uav_metrics/assessment.py','metrics/uav_metrics/vision.py','metrics/uav_metrics/learning.py','metrics/dataProfiler.py','metrics/featureAnalyzer.py','metrics/labelAnalyzer.py','metrics/IsoIecChecker.py'],
'ch09':['metrics/quality_aggregation.py','metrics/quality_profiles.py','metrics/config/adaptive_quality.json','metrics/experimental_scoring.py','metrics/config/experimental_score.json','backend/workbench/quality_api.py','backend/readiness_engine.py'],
'ch10':['backend/workbench/workflow_api.py','backend/workbench/lab_api.py','backend/workbench/experiments.py','backend/workbench/detector.py','backend/workbench/jobs.py','backend/workbench/worker.py','backend/workbench/splits.py','backend/workbench/variations.py','backend/workbench/combined.py','backend/research/experiments.py'],
'ch11':['backend/workbench/analytics.py','backend/workbench/api.py','backend/workbench/reporting.py'],
'ch12':['sql/models.py','sql/store.py','sql/datasets.py','backend/workbench/datasets.py','backend/workbench/detector.py','backend/workbench/jobs.py','main.py','backend/large_ingestion.py'],
'ch13':['main.py','backend/workbench/api.py','backend/workbench/jobs.py','backend/workbench/worker.py','backend/workbench/lab_api.py','backend/context_extractor.py','metrics/core.py'],
'ch14':['frontend/src/GuidedWorkflow.tsx','frontend/src/GuidedValidation.tsx','backend/workbench/workflow_api.py','backend/workbench/worker.py','backend/workbench/experiments.py','backend/workbench/detector.py','backend/workbench/analytics.py','backend/workbench/reporting.py','backend/workbench/synthetic.py'],
'ch15':[],
'ch16':['frontend/src/Workbench.tsx','backend/workbench/detector.py','backend/workbench/analytics.py','metrics/core.py','sql/store.py'],
'ch17':['metrics/IsoIecChecker.py','metrics/featureAnalyzer.py','metrics/uav_metrics/vision.py','metrics/uav_metrics/assessment.py','backend/workbench/analytics.py','backend/workbench/reporting.py','backend/research/__main__.py']}

DIAGRAM_CHAPTERS={'ch01':['system-overview'],'ch02':['dependency-core','dependency-workbench'],'ch03':['startup-sequence'],'ch04':['frontend-backend'],'ch06':['dataset-ingestion','large-data'],'ch07':['context-interpretation','object-lifecycle'],'ch08':['metrics-pipeline','metric-dispatch'],'ch09':['adaptive-aggregation'],'ch10':['split-safeguards','training-lifecycle','training-functions','experiment-data-flow','model-evaluation'],'ch11':['research-analytics'],'ch12':['storage-architecture','dataset-models'],'ch13':['error-recovery'],'ch15':['class-relationships'],'ch14':['reporting-flow']}

DIAGRAM_CHAPTERS['ch08'] += ['research-evidence','annotation-audit']
CHAPTER_SOURCES['ch04'] += ['frontend/src/ResearchEvidence.tsx','metrics/uav_metrics/presentation.py']
CHAPTER_SOURCES['ch08'] += ['metrics/uav_metrics/research.py','metrics/uav_metrics/pixels.py','metrics/uav_metrics/audit.py','metrics/uav_metrics/independence.py','metrics/uav_metrics/presentation.py']
CHAPTER_SOURCES['ch10'] += ['backend/research/variants.py','backend/workbench/contracts.py']
CHAPTER_SOURCES['ch11'] += ['metrics/uav_metrics/research.py','metrics/uav_metrics/audit.py','metrics/uav_metrics/independence.py']

def get_constant(path,name):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise ValueError(name)

def tree_text(paths):
    tree={}
    for path in sorted(paths):
        node=tree
        for part in path.split('/'):node=node.setdefault(part,{})
    lines=['C:/DataReadiness/']
    def visit(node,prefix=''):
        for i,(name,child) in enumerate(node.items()):
            last=i==len(node)-1
            lines.append(prefix+('`-- ' if last else '|-- ')+name+('/' if child else ''))
            if child:visit(child,prefix+('    ' if last else '|   '))
    visit(tree)
    return '\n'.join(lines)

def files_section():
    extras=['frontend/src/styles.css','frontend/src/workbench.css','frontend/src/workflow.css','frontend/package.json','frontend/vite.config.ts','frontend/vitest.config.ts','frontend/playwright.config.ts','frontend/tsconfig.json','metrics/config/adaptive_quality.json','metrics/config/experimental_score.json','requirements/requirements.txt','requirements/requirements-training.txt','requirements/requirements-dev.txt','pytest.ini','pyrightconfig.json']
    paths=sorted(set(FILES)|set(extras))
    out=['\n```text\n'+tree_text(paths)+'\n```\n','## File responsibilities and actual source dependencies\n']
    for group,title in [('', 'Root application entry'),('backend/', 'Backend ingestion and orchestration'),('metrics/', 'Metric calculations and scoring'),('sql/', 'Database schemas and persistence'),('backend/workbench/','Persistent workbench'),('metrics/uav_metrics/','UAV evidence'),('backend/research/','Standalone research'),('frontend/src/','Browser implementation'),('backend/scripts/','Repository tools'),('tests/','Python test suites')]:
        rows=[]
        for p,f in FILES.items():
            selected=p.startswith(group)
            if group=='':selected='/' not in p
            if group in ('backend/', 'metrics/', 'sql/'):
                selected=selected and '/' not in p[len(group):]
            if not selected:continue
            if group=='frontend/src/' and '.test.' in p:continue
            if group in ('tests/','backend/scripts/'):
                role=f.get('doc','').split('\n')[0] or ('Test assertions and fixtures.' if group=='tests/' else 'Standalone utility; inspect main/CLI before execution.')
            else:role=FILE_ROLES.get(p,f.get('doc','').split('\n')[0] or 'Package marker / shared namespace.')
            top=[s['name'] for s in f['symbols'] if '.' not in s['name']]
            dependency=[]
            for owner,g in FILES.items():
                module=p.removesuffix('.py').replace('/','.')
                if p.endswith('.py') and any(i['module']==module for i in g.get('imports',[]) if isinstance(i,dict)):dependency.append(owner)
            symbol_summary=(str(len(top))+' top-level tests/fixtures/helpers; all signatures are in chapter 15.') if group=='tests/' else '; '.join(top) or 'No declared top-level symbol'
            rows.append([link(p),role,symbol_summary, '; '.join(dependency[:5]) if dependency else 'No direct Python import found; may be an entry, route, TS import or dynamic use'])
        if rows:out+=['### '+title+'\n',table(['File','Responsibility','Main symbols','Direct Python importers (subset)'],rows)]
    out+=['### Configuration and styling\n',table(['File','Purpose'],[(link(p),FILE_ROLES.get(p,{'frontend/package.json':'Frontend scripts and package versions.','frontend/vite.config.ts':'Loopback dev host, API proxy and build chunking.','frontend/vitest.config.ts':'jsdom unit-test setup/exclusions.','frontend/playwright.config.ts':'Edge E2E browser, URL, viewport and output settings.','frontend/tsconfig.json':'Strict compile-time TypeScript configuration.','requirements/requirements.txt':'Base pinned Python dependencies.','requirements/requirements-training.txt':'Optional detector/torch packages plus base dependencies.','requirements/requirements-dev.txt':'Development typing dependencies plus base.','pytest.ini':'Python test discovery/runtime settings.','pyrightconfig.json':'Python analysis configuration.'}.get(p,'Repository configuration.'))) for p in extras])]
    return '\n'.join(out)

def route_consumer(path,handler,file):
    if file=='main.py':return 'App / useAssessment (original UI)' if path.startswith('/api/') and path not in ('/api/health','/api/config','/api/metrics') else 'Health, upload limits or metric-definition consumers'
    if file.endswith('workflow_api.py'):
        return {'provider_status':'GoalChat / SettingsPage','project_goal':'GoalChat.send','confirm_project':'GoalChat review button','assess_project':'ReadinessStep.run','prepare_validation':'GuidedValidation.prepare','validation_protocol':'GuidedValidation.protocol','get_validation':'GuidedValidation polling','reset_validation':'GuidedValidation reset buttons'}.get(handler,'GuidedWorkflow project create/list/restore')
    if file.endswith('quality_api.py'):return 'QualityAnalytics; AggregationField for history; registry API consumers'
    if file.endswith('lab_api.py'):return {'import_model':'ExperimentBuilder checkpoint import','external_benchmark':'PublishedBenchmarks','external_benchmarks':'PublishedBenchmarks','resume':'JobPanel resume button'}.get(handler,'ExperimentBuilder / GuidedValidation')
    return {'overview':'Overview','upload':'DatasetUpload / DatasetManager','directory_upload':'DatasetUpload / DatasetManager','register':'DatasetUpload / DatasetManager','datasets':'Shell / dataset selectors','dataset':'Dataset/library consumers','rename':'DatasetManager','delete':'DatasetManager','samples':'DatasetManager / sample viewer','image':'Boxes / img URLs','split':'TrainingPage; guided calls service via assess_project','splits':'SplitSelector / lab selection / ReadinessStep','context':'EvaluationPage','validate_context':'EvaluationPage','readiness':'EvaluationPage; guided assess_project','training':'TrainingPage','model_evaluation':'ComparisonPage','jobs':'Shell job polling','job':'Job consumers / diagnostics','cancel':'JobPanel','logs':'JobPanel','curves':'JobPanel','models':'Experiments / ComparisonPage / ValidationResults','artifact':'JobPanel checkpoint link','evaluations':'ComparisonPage / ValidationResults','predictions':'ComparisonPage','compare':'ComparisonPage audit','analytics':'Analytics (optional experiment_id scope)','variations':'Analytics controlled-data tool','report':'Report/Export links; original/saved/experiment scopes','score_preview':'ExperimentalScore','research_export':'Reports / advanced Analytics','settings':'SettingsPage','compute_status':'Shell / training device displays'}.get(handler,'Workbench API consumer')

def route_section():
    out=[]
    for file in ['main.py','backend/workbench/api.py','backend/workbench/workflow_api.py','backend/workbench/quality_api.py','backend/workbench/lab_api.py']:
        rows=[]
        prefix='' if file=='main.py' else '/api/v1/quality' if file.endswith('quality_api.py') else '/api/v1'
        for r in FILES[file]['routes']:
            dec=ast.parse(r['decorator'],mode='eval').body
            options={k.arg:ast.unparse(k.value) for k in dec.keywords}
            contract=r['signature'].replace('body:','body: ')
            response=options.get('response_model',r['response'] or 'Response / handler JSON; see body')
            calls=[x for x in r['calls'] if x not in ('str','len','sum','sorted','set','list','dict','bool','int','float','enumerate','isinstance','json.dumps','json.loads','print') and not x.startswith(('Path(',"__import__"))]
            flow='Calls: '+', '.join(calls[:7])+'. Consumer: '+route_consumer(prefix+r['path'],r['function'],file)+'.'
            rows.append([f'`{r["method"]} {prefix+r["path"]}`',link(file,r['function'])+'; input `'+contract+'`; output `'+response+'`; status '+options.get('status_code','200'),flow])
        out+=['### '+file+'\n',table(['Method / full path','Handler and actual boundary contract','Service calls and frontend consumer'],rows)]
    return '\n'.join(out)

def metric_section():
    diagnostic=get_constant('metrics/core.py','DIAGNOSTICS')
    out=['\n### Declared diagnostic execution order\n',' -> '.join(f'`{x}`' for x in diagnostic)+'\n']
    root=ast.parse((ROOT/'metrics/core.py').read_text(encoding='utf-8'))
    rows=[]
    for n in root.body:
        if not isinstance(n,ast.FunctionDef):continue
        for d in n.decorator_list:
            if isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='metric':
                values=[ast.literal_eval(v) for v in d.args[:3]]
                rows.append([values[0],link('metrics/core.py',n.name),values[1],values[2]])
    out+=['### Registered core checks (selection is separate from registration)\n',table(['Metric ID','Implementation','Dimension','Actual calculation declaration'],rows)]
    props=get_constant('metrics/uav_metrics/registry.py','PROPERTIES')
    enhancements=get_constant('metrics/uav_metrics/registry.py','ENHANCEMENTS')
    props=[(id,name,dimension,inputs,method+('; '+enhancements[id] if id in enhancements else ''),limits) for id,name,dimension,inputs,method,limits in props]
    out+=['### Stable UAV properties, in execution order\n',table(['ID / property','Required observations','Calculation / limitation'],[[id+' '+name,inputs,method+' '+limits] for id,name,dimension,inputs,method,limits in props])]
    return '\n'.join(out)

def policy_section():
    cfg=json.loads((ROOT/'metrics/config/adaptive_quality.json').read_text(encoding='utf-8'))
    dims=list(cfg['dimensions'])
    out=['\n### Exact task dimension ratios\n',table(['Task']+[cfg['dimensions'][d] for d in dims],[[name]+[p['dimension_weights'][d] for d in dims] for name,p in cfg['tasks'].items()])]
    out+=['### Components, sources and declared normalization\n',table(['Component / dimension','Evidence and normalization','Applicability / gate / limitation'],[[c['id']+' / '+c['dimension'],c.get('source','metric')+'; raw ID '+str(c.get('metric_id'))+'; '+c['normalization'], 'tasks '+str(c.get('tasks',[]))+'; modalities '+str(c.get('modalities',[]))+'; gate '+str(c.get('gate'))+'; '+c.get('limitations','')] for c in cfg['components']])]
    out+=['### Scenario policies and provenance\n',table(['Scenario','Compatible tasks','Explicit multipliers / explanation'],[[name,', '.join(p['tasks']),json.dumps({k:v for k,v in p.items() if k!='tasks'},ensure_ascii=True)] for name,p in cfg['scenarios'].items()])]
    return '\n'.join(out)

LINEAGE=[
('Raw upload','DatasetUpload / FastAPI UploadFile','binary stream + metadata','import_file -> load_dataset_stream','managed source / owned staging','Dataset'),
('Parsed samples','ingestion/image/record loaders','Dataset.frame; canonical objects','run_metric, subset, materialize','frame.json/evidence.json; DuckDB when large','metrics and detector inputs'),
('Dataset metadata','loader + persist','dataset RecordResponse payload','selectors, split/readiness/training','research_records kind=dataset','version/provenance and source refs'),
('Task tags','App / EvaluationPage','dict[str,list[str]]','interpret','DatasetContext.selected_tags','structured draft; guided path uses empty tags'),
('User context','interpret / confirm_project','DatasetContext containing Goal','readiness -> worker -> assess','context records and job config snapshots','Certificate.confirmed_context'),
('Requirement','build_requirements + manual Goal requirements','Requirement','run_metric + evaluate','inside Certificate/context JSON','AssessmentFinding'),
('Raw measurement','run_metric implementation','MetricResult + AssessmentEvidence','evaluate, learning views, aggregation','completed job.result.metrics','findings and displayed evidence'),
('Cached pixel evidence','inspect_image / collect_measurements','frame columns; histogram/count dictionaries','clarity/visibility normalization and charts','frame/evidence and Certificate.quality_measurements','observed bins and criterion fractions'),
('Policy','quality_profiles.load_profile','QualityProfile','aggregate_quality','configuration plus full frozen profile in result','effective weights, versions and hashes'),
('Normalized criteria','quality_aggregation._measure','component row dict with score/status/evidence','within-dimension aggregation and gates','quality_profile.individual_metrics/dimensions','supported scores or null'),
('Readiness certificate','assess/_assess','Certificate','reporting, model provenance, research','completed readiness job.result','JSON/HTML/CSV/PDF and UI'),
('Split manifest','create_split / _matched_split','record with images/hashes/groups/checksum','subset, validation_source, test_manifest','split record','exact train/val/test membership'),
('Training configuration','TrainRequest / ExperimentRequest','validated model then config dict','worker -> train / resume_training','job config and configuration.json','reproducible model procedure'),
('Experiment','create_experiment','request + protocol + IDs/status','start_experiment -> execute_experiment','experiment record','models and final evaluations'),
('Checkpoint','YOLO.train / model.save','trusted .pt file','finish_training, resume, evaluate_models','jobs/id/run/weights or managed import','model artifact hash and predictions'),
('Predictions','YOLO.predict + repository matching','per-image truth/predictions/missed dicts','viewer, metrics, confusion aggregation','predictions.json referenced by evaluation','actual same-image visualization'),
('Evaluation','evaluate_models','evaluation payload with metrics/settings/curves','comparison, research, BenchmarkCharts','evaluation record','held-out model evidence'),
('Research analysis','analytics.research','rows/aggregates/correlations/pairs dict','Analytics + reporting','calculated on read; export may save analysis','exploratory scoped report'),
('Readable application report','reporting.document / reports.html_report','document dict or escaped HTML','HTML/PDF renderer / download','response bytes; analysis snapshot when requested','human-readable evidence export'),
]

LINEAGE += [
('New global pixel cache','inspect_image -> pixel_features','protected research_* columns and hexadecimal dhash64','UAV-12/14/18, collect_measurements','frame.json; no new SQL table','global degradation and duplicate evidence'),
('Runtime image reader','datasets.load / combined_readiness','in-process callable returning image_source context manager','pixels.source_reader -> bounded local inspection','not serialized; reconstructed on load','verified access to selected members'),
('Scoped independent audit','dataset metadata -> subset','complete reference images, protocol and parent/subset fingerprints','annotation_audit / negative_coverage','evidence.json and metric values','audited subset categories and meaningful denominators'),
('Local pixel observations','local_measurements','bounded per-object scratch records and shared session summary','UAV-12/14','assessment-local cache; summaries in MetricResult','sampled distributions, class/size strata and candidates'),
('Joint scenario evidence','scenario_evidence / scenarios','matching/nonmatching/unknown image and object counts','scenario_count, UAV-17, confirmed requirements','MetricResult.value / evidence.details','lower-bound count checks and reference comparisons'),
('Duplicate candidate components','collection_evidence -> HashTree.find_bounded','bounded Hamming candidates plus declared groups','UAV-18 / research evidence views','MetricResult.value','candidate groups, leakage warnings and incomplete-retrieval scope'),
('Research presentation views','evidence_sections','eight sections derived from serialized MetricResults','ResearchEvidence, HTML/PDF rendering','Certificate.quality_measurements.research_evidence','charts/tables with actual units, sample sizes and missing states'),
('Native-size operating recall','evaluate_models -> native_size_outcomes','ground-truth area counts, missed truth and separately binned FP','evaluation metrics and research outcome selector','evaluation.size_stratified and metric fields','real native-area recall; no invented small-object AP'),
]

def deep_section():
    out=['\n## Architecturally important function cards\n']
    for d in FUNCTIONS:
        s=SYMS[(d['path'],d['name'])]
        out+=[anchor('fn-'+re.sub(r'[^a-z0-9]+','-',d['path']+'-'+d['name'],flags=re.I).lower()),'### Function: `'+d['name']+'()`\n',
            '**Location and source:** '+link(d['path'],d['name'])+'\n',
            '**Actual declaration:** `'+d['name']+s['signature']+(' -> '+s['returns'] if s.get('returns') else '')+'`\n']
        for key,label in [('purpose','Purpose'),('callers','Who calls it?'),('receives','What does it receive?'),('steps','What happens internally?'),('returns','What does it return?'),('consumers','Who receives the output?'),('effects','Side effects'),('failures','Failure conditions'),('analogy','Simple analogy')]:out+=['**'+label+':** '+d[key]+'\n']
        out+=['**Related diagram:** [Figure: '+next(x['title'] for x in DIAGRAMS if x['id']==d['diagram'])+'](#diagram-'+d['diagram']+')\n']
    return '\n'.join(out)

def reference_section():
    out=['\n## All inspected Python symbols\n']
    for file in PY['python']:
        out+=['### Source reference: '+file['path'].replace('_','\\_')+'\n']
        if not file['symbols']:out+=['Package marker; no declared functions or classes.\n'];continue
        rows=[]
        for s in file['symbols']:
            doc=(s['doc'].split('\n')[0] if s['doc'] else ('Test/fixture/helper declaration; inspect assertions for behavior.' if file['path'].startswith('tests/') else 'Internal or public declaration; see source and selected workflow cards.'))
            calls=', '.join(s['calls'][:5])
            rows.append([link(file['path'],s['name'])+'; '+s['kind'],'`'+s['signature']+'`'+(' -> `'+s['returns']+'`' if s['returns'] else '; return not annotated'),doc+(' Syntactic calls: '+calls+'.' if calls else '')])
        out+=[table(['Symbol / kind','Actual signature / declared return','Description / syntactic evidence'],rows)]
    out+=['## All inspected TypeScript declarations\n']
    for file in TS:
        out+=['### Source reference: '+file['path'].replace('_','\\_')+'\n']
        if not file['symbols']:out+=['No named declarations extracted; may contain module-level tests/setup.\n'];continue
        rows=[]
        for s in file['symbols']:
            rows.append([link(file['path'],s['name']),s['kind'],'`'+s['signature']+'`; return/type `'+s['returns']+'`'])
        out+=[table(['Symbol','Declaration kind','Actual parameters / type body / return'],rows)]
    out+=['## Class fields and model lifecycles\n','This table lists declared annotated fields, without silently expanding arbitrary dictionaries into invented schemas. Pydantic defaults and dataclass field factories are actual source expressions. SQLAlchemy columns use assignments rather than annotations and are described in chapter 12.\n']
    for file in PY['python']:
        if file['path'].startswith(('tests/','backend/scripts/')):continue
        for s in file['symbols']:
            if s['kind']!='class' or not s['fields']:continue
            out+=['### Fields: '+file['path']+' / '+s['name']+'\n',table(['Field','Declared type','Default expression'],[[x['name'],'`'+x['type']+'`','`'+x['default']+'`' if x['default'] else 'required / no source default'] for x in s['fields']])]
    return '\n'.join(out)

def test_section():
    rows=[]
    for f in PY['python']+TS:
        p=f['path']
        if p.startswith('tests/') or '.test.' in p or p.startswith('frontend/e2e/'):
            names=[s['name'] for s in f['symbols'] if s['name'].split('.')[-1].startswith('test_')]
            rows.append([link(p),f.get('doc','').split('\n')[0] or ('Browser integration checks.' if '/e2e/' in p else 'React behavior/contracts tests.'),str(len(names))+' named Python test functions' if p.endswith('.py') else 'it/test callbacks; inspect the suite for cases'])
    out=['\n## Tests and external dependency boundaries\n',table(['Suite','Evidence area','Inventory note'],rows)]
    deps=[]
    for file in ['requirements/requirements.txt','requirements/requirements-training.txt','requirements/requirements-dev.txt']:
        deps += [[link(file),line] for line in (ROOT/file).read_text().splitlines() if line.strip() and not line.startswith(('#','-r'))]
    out+=['### Declared Python dependencies\n',table(['Declaration file','Pinned dependency'],deps)]
    pkg=json.loads((ROOT/'frontend/package.json').read_text())
    out+=['### Declared frontend dependencies\n',table(['Group','Package','Version range'],[[group,k,v] for group in ['dependencies','devDependencies'] for k,v in pkg[group].items()])]
    out+=['### Environment configuration names (not current values)\n']
    env={}
    for p in [x['path'] for x in PY['python'] if not x['path'].startswith(('tests/','backend/scripts/'))]:
        tree=ast.parse((ROOT/p).read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='getenv' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):env.setdefault(n.args[0].value,set()).add(p)
            if isinstance(n,ast.Subscript) and ast.unparse(n.value)=='os.environ' and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str):env.setdefault(n.slice.value,set()).add(p)
    out+=[table(['Name used by code','Owning files'],[[name,', '.join(link(p) for p in sorted(paths))] for name,paths in sorted(env.items())]),'No environment-file values or current process secret values were read to build this table. Presence/usage in source does not mean a setting is active in a particular user session.\n']
    return '\n'.join(out)

def excerpt(path,name,limit):
    s=SYMS[(path,name)]
    lines=(ROOT/path).read_text(encoding='utf-8-sig').splitlines()
    return '\n### Source excerpt: '+name+'\n\n'+link(path,name)+'\n\n```'+('typescript' if path.endswith(('.ts','.tsx')) else 'python')+'\n'+'\n'.join(lines[s['line']-1:min(s['end'] if 'end' in s else s['line']+limit,s['line']+limit-1)])+'\n```\n'

def build():
    out=['# FlightReady: a visual codebase textbook\n','**A beginner-friendly, source-derived guide to the UAV Dataset Readiness and Model Validation platform.**\n',
         'Snapshot: '+SNAPSHOT+'. This edition incorporates the research-evidence-2.0 pipeline and engine 1.5.0. Analysis covers '+str(len(PY['python']))+' Python files and '+str(len(TS))+' TypeScript files, '+str(sum(len(f['symbols']) for f in PY['python']))+' Python symbols and '+str(sum(len(f['symbols']) for f in TS))+' TypeScript declarations, including tests. Source inventory excludes dependencies, runtime uploads, databases and checkpoints.\n',
         anchor('contents'),'## Table of contents\n']
    out += ['- ['+title+'](#'+id+')' for id,title in CHAPTERS]
    out+=['\n'+anchor('understand'),'## I Want to Understand...\n']
    quick=[('How the website communicates with Python','ch04'),('How a dataset is uploaded and normalized','ch06'),('How natural-language requirements become a goal','ch07'),('How a metric is calculated','ch08'),('How weights are selected and why some matter more','ch09'),('How score masking and unknown evidence are handled','ch09'),('How real model training starts and uses the GPU','ch10'),('How model performance and fair comparisons work','ch10'),('How quality and downstream performance are connected','ch11'),('How results are stored','ch12'),('How a report is generated','diagram-reporting-flow'),('How errors and recovery work','ch13'),('Which function receives and returns each object','ch15'),('Where to debug or extend a component','ch16')]
    out += ['- ['+label+'](#'+id+')' for label,id in quick]
    for id,title in CHAPTERS:
        out+=['\n---\n',anchor(id),'# '+title+'\n',TEXT[id].strip()+'\n']
        if id=='ch02':out+=[files_section()]
        if id=='ch05':out+=[route_section()]
        if id=='ch08':out+=[metric_section()]
        if id=='ch09':out+=[policy_section()]
        if id=='ch12':out+=['\n## Complete data-lineage reference\n',table(['Data object','Created by','Type / schema','Consumed by','Stored where','Final output'],LINEAGE)]
        if id=='ch15':out+=[deep_section(),reference_section()]
        if id=='ch16':out+=['\n## Practical debugging roadmap\n',table(['Symptom','Trace / what to inspect','Source files','Relevant tests/tools'],[[symptom,trace,'; '.join(link(p.strip()) for p in files.split(';')),'; '.join(link(p.strip()) for p in tests.split(';'))] for symptom,trace,files,tests in DEBUG])]
        if id=='ch17':out+=['\n## Glossary\n',table(['Term','Meaning in this application'],GLOSSARY),test_section()]
        for d in DIAGRAM_CHAPTERS.get(id,[]):out+=[diagram(d)]
        if id=='ch04':out+=[excerpt('frontend/src/api.ts','api',7)]
        if id=='ch09':out+=[excerpt('metrics/quality_aggregation.py','geometric',12)]
        if id=='ch12':out+=[excerpt('sql/store.py','Store.create',7)]
        if id=='ch13':out+=[excerpt('backend/workbench/worker.py','run',11)]
        sources=CHAPTER_SOURCES[id]
        if id!='ch02' and sources:out+=['\n**Chapter evidence:** '+', '.join(link(p) for p in sources)+'.\n']
        out+=['\n[Back to contents](#contents) | [Question map](#understand)\n']
    out+=['\n'+anchor('verification-map'),'# Source verification and diagram index\n',table(['Figure','Type','Supporting source files','What was verified'],[[f'[{d["title"]}](#diagram-{d["id"]})',d['kind'],', '.join(link(p) for p in d['files']),'Curated against inspected source; Mermaid rendered and checked by the export verification report.'] for d in DIAGRAMS]),
          '\nThe accompanying verification report records actual export validation counts. A diagram is not claimed rendered solely because its source fence exists. SVG/PNG files are generated from these exact Mermaid blocks. HTML and PDF exporters read this same Markdown and those assets, without independently rewriting the architectural explanations.\n']
    markdown='\n'.join(out).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    (DOC/'CODEBASE_VISUAL_GUIDE.md').write_text(markdown,encoding='utf-8')
    (ASSETS/'diagrams.json').write_text(json.dumps(DIAGRAMS,indent=2),encoding='utf-8')
    manifest={'python_files':len(PY['python']),'typescript_files':len(TS),'python_symbols':sum(len(f['symbols']) for f in PY['python']),'typescript_symbols':sum(len(f['symbols']) for f in TS),'route_declarations':sum(len(f['routes']) for f in PY['python']),'deep_function_cards':len(FUNCTIONS),'diagrams':len(DIAGRAMS),'chapters':len(CHAPTERS),'characters':len(markdown),'words':len(markdown.split())}
    (ASSETS/'guide-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))

if __name__=='__main__':build()
