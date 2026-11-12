"""Pipeline diagrams updated for the current source; stable figure anchors remain."""

def refresh(diagrams, add):
    def replace(id, source, explanation=None, files=None):
        d=next(d for d in diagrams if d['id']==id)
        d['source']=source.strip()
        if explanation:d['explanation']=explanation
        if files:d['files']=files

    replace('metrics-pipeline',r'''
flowchart TD
  A["Confirmed context + exact Dataset population"] --> B["build_requirements: explicit task criteria"]
  B --> C["run_metric + evaluate requirements in order"]
  C --> D["DIAGNOSTICS then selected UAV-01..20"]
  D --> E["Enhanced geometry, local pixels, audit,<br/>negative, scenario and group evidence"]
  E --> F["learning_views + recommendations"]
  F --> G["modality findings + decide requirements"]
  G --> H["retained experimental scalar"]
  H --> I["Certificate + engine/metric fingerprints"]
  I --> J["collect_measurements: cached global distributions"]
  J --> P["evidence_sections: eight stored presentation views"]
  P --> K["aggregate_quality: existing six dimensions and gates"]
  K --> S[("completed readiness job")]
  S --> U["guided/advanced ResearchEvidence + exports"]
''','The requirement verdict, raw observations, readable evidence views and adaptive status remain separate outputs. The existing assessment session shares bounded local image work; adding a view does not add a new scoring engine or trigger model training.',['backend/readiness_engine.py','metrics/core.py','metrics/uav_metrics/assessment.py','metrics/uav_metrics/presentation.py','metrics/quality_aggregation.py','frontend/src/ResearchEvidence.tsx'])

    replace('metric-dispatch',r'''
flowchart TD
  A["run_metric: ID + Dataset + Goal + parameters"] --> B{"Registered / modality supported?"}
  B -->|"no"| N["unavailable or not_applicable"]
  B -->|"yes"| C{"UAV property?"}
  C -->|"yes"| U["assessment.calculate under stable UAV ID"]
  U --> X{"output_path requested?"}
  X -->|"yes"| Y["reject unavailable parent/leaf;<br/>preserve measured or estimated provenance"]
  X -->|"no"| R["MetricResult with structured evidence"]
  Y --> R
  C -->|"no"| D{"disk_path present?"}
  D -->|"yes"| SQL["whole-data disk metric"]
  D -->|"no"| P["registered pandas/image function;<br/>scenario_count shares scenario_evidence"]
  SQL --> R
  P --> R
  N --> R
  R --> E["evaluate explicit requirement or retain diagnostic"]
''','A structured count is usable only with its actual measurement availability and denominator. Unknown local/audit evidence cannot be projected into a zero quality value. Scenario checks reuse the shared three-valued evaluator, including count lower bounds.',['metrics/core.py','backend/readiness_engine.py','metrics/uav_metrics/research.py','metrics/uav_metrics/assessment.py'])

    replace('model-evaluation',r'''
flowchart TD
  A["evaluate_models: model IDs + independent-test config"] --> B["test_manifest + classes + assert_independent"]
  B --> C["materialize verified canonical test data"]
  C --> D["YOLO.val: AP confidence sweep"]
  D -->|"mAP + per-class AP + curves"| R["actual evaluation metrics"]
  C --> E["YOLO.predict: declared operating confidence"]
  E --> F["detection_errors: class-aware one-to-one matching"]
  E --> G["confusion_counts: spatial class-agnostic matching"]
  F -->|"TP / FP / FN and defined ratios"| R
  F --> N["native_size_outcomes:<br/>truth-area recall; FP by predicted area"]
  N -->|"small / medium / large native recall"| R
  G -->|"predicted rows / true columns"| R
  R --> S[("evaluation + predictions.json + protocol version")]
  S --> U["BenchmarkCharts / comparison / research outcomes"]
''','Size-stratified recall uses ground-truth native area and the actual operating matching results. False positives have no true-object size and are binned separately by predicted area. Small-object AP remains unavailable; this is not official COCO area evaluation.',['backend/workbench/detector.py','backend/workbench/analytics.py','frontend/src/Workbench.tsx','frontend/src/BenchmarkCharts.tsx'])

    replace('reporting-flow',r'''
flowchart TD
  C[("saved Certificate: raw metrics + quality_measurements")]
  C --> V["research_evidence: eight derived views"]
  V --> UI["QualityAnalytics -> ResearchEvidence"]
  U["export selected report ID + format"] --> A["workbench.api.report"]
  A --> D{"Record / format?"}
  D -->|"readiness JSON / CSV"| J["complete saved record / structured metric rows"]
  D -->|"readiness HTML"| H["reports.html_report + research_evidence_html"]
  D -->|"PDF / research HTML"| R["reporting.document: exact saved scope"]
  C --> H
  C --> R
  R --> P["pdf_document / html_document:<br/>evidence tables, units, sampling and limitations"]
  J --> O["download response"]
  H --> O
  P --> O
''','The UI and exports consume saved backend evidence. They do not create substitute measurements or empirical conclusions. Application PDF methodology/raw blocks may be explicitly abbreviated with JSON as the full raw export; this textbook exporter separately preserves the entire guide.',['backend/workbench/api.py','backend/workbench/reporting.py','backend/reports.py','metrics/uav_metrics/presentation.py','frontend/src/ResearchEvidence.tsx'])

    models=next(d for d in diagrams if d['id']=='dataset-models')
    models['source']=models['source'].replace('    disk_path : str or None','    disk_path : str or None\n    image_reader : runtime callable or None')
    models['explanation']+=' The image_reader is a runtime-only capability reconstructed on load; persist does not serialize executable access into evidence JSON.'
    ingestion=next(d for d in diagrams if d['id']=='dataset-ingestion')
    ingestion['explanation']+=' Inspection now caches dhash64 and global research_* proxies. Scoped dataset metadata preserves independent audit provenance; local crop/ring inspection is a later bounded assessment operation.'

    add('research-evidence','The v2 instruments share one assessment and one evidence ledger','flowchart',r'''
flowchart TD
  A["ingested frame + boxes + hashes + declared metadata"] --> S["exact population: whole data or scoped training subset"]
  S --> D["existing assessment.calculate dispatcher"]
  D --> G["research.scale_analysis -> UAV-08"]
  D --> P["pixels.local_measurements -> UAV-12 / UAV-14"]
  D --> Q["audit.annotation_audit / negative_coverage -> UAV-15"]
  D --> C["research.scenarios -> UAV-17"]
  D --> I["independence.collection_evidence -> UAV-18"]
  S --> F["cached global pixels / dhash64"]
  F --> P
  F --> I
  C --> E["shared scenario_evidence"]
  R["requirements: scenario_count"] --> E
  P --> K["one session result;<br/>bounded verified image reads"]
  Q --> V["reference trust/scope + separate category denominators"]
  I --> B["capped HashTree.find_bounded;<br/>completeness retained"]
  G --> M["MetricResult / AssessmentEvidence"]
  K --> M
  V --> M
  E --> M
  B --> M
  M --> T["Certificate + readable evidence views"]
''','The new modules are instruments inside the stable registry. Exact native geometry, sampled pixels, declared metadata, independent references and incomplete duplicate retrieval retain different provenance and denominators. None is automatically a model-performance predictor or a new primary score.',['metrics/uav_metrics/research.py','metrics/uav_metrics/pixels.py','metrics/uav_metrics/audit.py','metrics/uav_metrics/independence.py','metrics/uav_metrics/assessment.py','metrics/uav_metrics/presentation.py'])

    add('annotation-audit','Independent audit: scope and one-to-one matching before error rates','flowchart',r'''
flowchart TD
  A["declared human / trusted external references"] --> T{"Independent identity, protocol,<br/>complete references and valid scope?"}
  T -->|"no"| U["semantic correctness unverified"]
  T -->|"yes"| S["training subset: filter references;<br/>retain parent audit fingerprint"]
  S --> B["validate normalized boxes + resource caps"]
  B --> X["separate missing/invalid syntax<br/>and ignored/uncertain regions"]
  X --> H["descending high-IoU one-to-one matching"]
  H --> W["unmatched high-IoU class discrepancies"]
  W --> L["lower-IoU same-class localization matching"]
  L --> F["unmatched references: missing objects"]
  L --> P["unmatched annotations:<br/>duplicates or spurious boxes"]
  X --> N["explicit incorrect negatives / syntax categories"]
  F --> R["separate reference-object, annotation<br/>and image denominators"]
  P --> R
  N --> R
  R --> C["class/size strata + audited coverage<br/>and limitations; no subset extrapolation"]
  C --> M["UAV-15 structured_audit / negative_coverage"]
''','The source labels remain unchanged. Training-set model predictions are not independent reference truth. Syntax failure does not become a semantic missing-object count, and zero denominators remain null. The new category audit does not silently replace the older count-only agreement score.',['metrics/uav_metrics/audit.py','backend/workbench/datasets.py','metrics/uav_metrics/assessment.py','metrics/quality_aggregation.py'])
