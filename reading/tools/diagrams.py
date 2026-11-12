"""Curated Mermaid sources. Exports render these exact strings from Markdown."""
DIAGRAMS = []

def add(id, title, kind, source, explanation, files):
    DIAGRAMS.append(dict(id=id,title=title,kind=kind,source=(source.replace(';',',') if kind=='sequence' else source).strip(),explanation=explanation,files=files))

add('system-overview','The complete system: a dataset research laboratory','flowchart',r'''
flowchart TD
  B["Browser: Workbench / GuidedWorkflow"] -->|"files, goal, actions"| A["main.py / workbench API routers"]
  A -->|"bytes"| I["datasets.import_file"]
  A -->|"task text only"| C["context_extractor.interpret"]
  A -->|"task/config"| J["JobManager.enqueue"]
  J -->|"separate process"| W["worker.run"]
  W -->|"readiness"| M["readiness_engine.assess<br/>metrics + adaptive scoring"]
  W -->|"experiment/training/test"| T["experiments / detector"]
  I -->|"Dataset evidence"| S[("Store: research_records")]
  C -->|"context after review"| S
  M -->|"Certificate"| S
  T -->|"models, evaluations"| S
  S -->|"saved evidence"| R["analytics / reporting"]
  R -->|"JSON, HTML, CSV, PDF"| O["Browser results / downloads"]
''','The trigger is a user submitting data or a confirmed goal. HTTP accepts work; JobManager crosses a process boundary for compute. SQLite holds JSON records; checkpoints and datasets remain on disk. The context provider is a separate optional network path, not the training engine.',['main.py','backend/workbench/api.py','backend/workbench/worker.py','backend/workbench/reporting.py'])

add('startup-sequence','Startup and the first browser visit','sequence',r'''
sequenceDiagram
  participant CLI as uvicorn main:app
  participant App as main.lifespan
  participant Services as api.services
  participant DB as Store
  participant Queue as JobManager
  participant UI as Browser / Shell
  CLI->>App: enter FastAPI lifespan
  App->>Services: services() cached singleton
  Services->>DB: Store(); create_all + WAL
  Services-->>App: (store, manager)
  App->>Queue: recover(); start()
  UI->>CLI: GET / and built assets
  UI->>CLI: GET /api/v1/datasets, jobs, compute
  UI->>CLI: GET /api/v1/workflows
  CLI-->>UI: stored records / actual compute state
  Note over Queue,UI: Jobs are polled; no WebSocket or server event stream
''','Starting the HTTP server creates the persistent services through lifespan. A production frontend is served only if frontend/dist existed when main.py imported. In development, Vite forwards /api requests instead.',['main.py','backend/workbench/api.py','backend/workbench/jobs.py','frontend/src/Workbench.tsx'])

add('frontend-backend','A browser action becomes a durable goal','sequence',r'''
sequenceDiagram
  participant User
  participant Chat as GoalChat.send
  participant Client as api.ts api / json
  participant Route as workflow_api.project_goal
  participant Extract as interpret
  participant Store
  User->>Chat: submit a goal message
  Chat->>Client: POST /workflows/{id}/goal
  Client->>Route: MessageRequest JSON
  Route->>Store: load workflow; require_idle
  Route->>Extract: accumulated user text; allow_remote
  Extract-->>Route: unconfirmed DatasetContext
  Route->>Store: create context; update workflow pointers
  Route-->>Client: RecordResponse with draft context
  Client-->>Chat: query cache update; render review
  Note over User,Store: confirm_project is a separate request; interpretation does not confirm
''','The JSON helper sets headers and serialization. The server saves the draft before returning it. GoalChat updates React Query, but the source of truth for saved goals is the server record.',['frontend/src/GuidedWorkflow.tsx','frontend/src/api.ts','backend/workbench/workflow_api.py','backend/context_extractor.py'])

add('dependency-core','Core imports: ingestion, metrics and decisions','flowchart',r'''
flowchart TD
  A["backend/readiness_engine.py"] -.->|"imports"| B["backend/context.py: contracts"]
  A -.->|"imports"| C["metrics/core.py: run_metric"]
  A -.->|"local import"| D["metrics/quality_aggregation.py"]
  C -.->|"imports Dataset"| E["backend/ingestion.py"]
  C -.->|"register_uav"| F["metrics/uav_metrics/registry.py"]
  F -.->|"lazy calculation import"| G["metrics/uav_metrics/assessment.py"]
  D -.->|"policy models"| H["metrics/quality_profiles.py"]
  E -.->|"large / general formats"| I["backend/large_ingestion.py"]
  I -.->|"archive dispatch"| J["backend/dataset_archives.py"]
''','Every dashed edge here means an import dependency, not a claim that the caller executes the imported function on every request. Lazy imports and the metric registry mediate several runtime paths.',['backend/readiness_engine.py','metrics/core.py','metrics/quality_aggregation.py','backend/ingestion.py','backend/large_ingestion.py'])

add('dependency-workbench','Workbench imports and runtime boundaries','flowchart',r'''
flowchart TD
  A["main.py"] -.->|"register routers"| B["api / workflow_api / lab_api / quality_api"]
  B -.->|"cached services"| C["Store + JobManager"]
  C -->|"python -m backend.workbench.worker"| D["worker.run in child process"]
  D -.->|"lazy task implementation"| E["detector / experiments"]
  E -.->|"source validation"| F["datasets / splits"]
  E -.->|"optional ML library"| G["Ultralytics / PyTorch"]
  B -.->|"read/export"| H["analytics / reporting"]
  H -.->|"record access"| C
''','Dashed arrows mean imports. The solid arrow is process creation at runtime. The worker is a subprocess entry point, not a FastAPI endpoint or a second web server.',['main.py','backend/workbench/api.py','backend/workbench/jobs.py','backend/workbench/worker.py','backend/workbench/detector.py'])

add('dataset-ingestion','Uploaded bytes become a normalized Dataset','flowchart',r'''
flowchart TD
  A["workbench.api.upload: UploadFile"] -->|"file handle + metadata"| B["datasets.import_file"]
  B -->|"bounded stream"| C["large_ingestion.load_dataset_stream"]
  C --> D{"Source format?"}
  D -->|"container"| E["dataset_archives.load_container"]
  E -->|"ambiguous layout"| X["SelectionRequired: explicit parser choice"]
  E -->|"image layout"| F["load_image_archive / load_universal_images"]
  D -->|"CSV / Parquet / record formats"| G["pandas or DuckDB table loading"]
  F -->|"frame + objects + hashes"| H["ingestion.Dataset"]
  G -->|"full profile; bounded preview if disk-backed"| H
  H -->|"frame.json + evidence.json + optional table.duckdb"| I["datasets.persist"]
  I -->|"metadata JSON"| S[("Store dataset record")]
''','Intake normalizes data without running uploaded source code. The legacy /api route exposes a selection endpoint for ambiguous archives; the persistent workbench requires parser options on import and does not offer that same selection wizard.',['backend/workbench/api.py','backend/workbench/datasets.py','backend/large_ingestion.py','backend/dataset_archives.py','backend/universal_images.py'])

add('large-data','Large tables: preview and full-data checks are different','flowchart',r'''
flowchart TD
  A["load_dataset_stream: staged upload"] -->|"CSV batches / Parquet batches / records"| B["DuckDB private table"]
  B -->|"whole-table SQL"| C["full_profile"]
  B -->|"LIMIT 100 preview"| P["Dataset.frame"]
  B -->|"database path"| D["Dataset.disk_path"]
  C -->|"global statistics"| E["Dataset.profile"]
  D -->|"run_metric detects disk backend"| F["disk_metrics.run_disk_metric"]
  P -->|"display only"| UI["profile / sample UI"]
  F -->|"full-table result or explicit unsupported"| M["MetricResult"]
''','The browser preview is not the statistical population. Disk metrics query the stored table; UAV properties that lack a full-data disk implementation report unsupported evidence rather than scoring the preview.',['backend/large_ingestion.py','metrics/disk_metrics.py','metrics/core.py','metrics/uav_metrics/assessment.py'])

add('context-interpretation','Text interpretation, validation and explicit confirmation','flowchart',r'''
flowchart TD
  U["GoalChat: user goal messages"] --> A["workflow_api.project_goal"]
  A -->|"text + empty tags; allow_remote flag"| I["context_extractor.interpret"]
  I --> L["deterministic fallback draft"]
  I --> D{"Configured provider enabled?"}
  D -->|"Gemini / compatible REST"| R["provider.extract: task text only"]
  R --> V["DatasetContext validation + numeric grounding"]
  V -->|"valid"| S["unconfirmed saved context"]
  V -->|"unavailable / invalid"| L
  L --> S
  S -->|"human reviews fields/questions"| C["confirm_project"]
  C -->|"new confirmed context record"| F["workflow.confirmed_context_id"]
  F -->|"exact context snapshot"| Q["assess_project -> readiness"]
''','There is no automatic confirmation by a model. The fallback remains explicit, clarification questions can block analysis, and external responses must pass the local schema. No uploaded dataset files are sent to the interpreter.',['backend/context_extractor.py','backend/context.py','backend/workbench/workflow_api.py','frontend/src/GuidedWorkflow.tsx'])

add('metrics-pipeline','Actual order inside readiness_engine._assess','flowchart',r'''
flowchart TD
  A["Confirmed DatasetContext + Dataset"] --> B["build_requirements(goal)"]
  B -->|"ordered Requirement list"| C["run_metric + evaluate for each requirement"]
  C --> D["DIAGNOSTICS in declared order"]
  D -->|"image/UAV/telemetry context"| E["append PROPERTY_IDS UAV-01..UAV-20"]
  E --> F["learning_views + recommendations"]
  F --> G["modality findings + decide"]
  G --> H["experimental_scoring.calculate"]
  H --> I["Certificate + reproducibility fingerprint"]
  I --> J["collect_measurements from cached pixels"]
  J -->|"supported task"| K["aggregate_quality + final fingerprint"]
''','The engine runs requirement checks first and then diagnostics. A metric can therefore be called more than once with different parameters. This is a deterministic ordered pipeline, not parallel metric voting.',['backend/readiness_engine.py','metrics/core.py','metrics/uav_metrics/registry.py','metrics/quality_aggregation.py'])

add('metric-dispatch','run_metric: one public dispatcher, several implementations','flowchart',r'''
flowchart TD
  A["metrics.run_metric(id, ds, goal, parameters, reference)"] --> B{"Registered and supported modality?"}
  B -->|"no"| N["unknown / not_applicable result"]
  B -->|"yes"| C{"UAV property ID?"}
  C -->|"yes"| U["CALCULATIONS[id] -> assessment.calculate"]
  C -->|"no"| D{"Dataset.disk_path present?"}
  D -->|"yes"| SQL["disk_metrics.run_disk_metric"]
  D -->|"no"| P["CALCULATIONS[id]: pandas/image check"]
  U -->|"optional output_path extraction"| R["MetricResult"]
  SQL --> R
  P --> R
  N --> R
  R -->|"state/value/evidence"| E["evaluate requirement or display diagnostic"]
''','Registry membership selects an implementation; it does not automatically make a metric a hard requirement. MissingEvidence and calculation exceptions are converted into explicit result states.',['metrics/core.py','metrics/disk_metrics.py','metrics/uav_metrics/assessment.py'])

add('adaptive-aggregation','Raw evidence becomes six independent quality dimensions','flowchart',r'''
flowchart TD
  A["Certificate metrics/findings + quality_measurements"] --> B["aggregate_quality: copy frozen inputs"]
  P["QualityProfile + task/scenario + confirmed overrides"] --> B
  B --> C["_measure: applicability + normalized component score"]
  C --> D["effective component weights; unknowns retain weight"]
  D --> E{"Weighted coverage equals 1?"}
  E -->|"no"| N["dimension score = null"]
  E -->|"yes"| G["geometric: zero preserved"]
  C -->|"separate evaluation"| K["critical gates + confirmed findings"]
  N --> S["six scores/statuses + normalized importance"]
  G --> S
  K --> R["adaptive readiness_status"]
  S --> R
  R --> O["profile + versions + hashes + evidence"]
''','The weights combine components within a dimension. Dimension importance never collapses the six dimensions into a primary percentage. A critical failure survives even when its score weight is zero.',['metrics/quality_aggregation.py','metrics/quality_profiles.py','metrics/config/adaptive_quality.json'])

add('class-relationships','Context and certificate contracts: inheritance and composition','class',r'''
classDiagram
  class Model {
    extra forbidden
    finite numbers
  }
  class Goal {
    task_type
    required_classes
    scenarios
    requirements
    quality_policy
  }
  class DatasetContext {
    original_text
    goal
    clarification_questions
    provider
    confirmed
  }
  class Requirement {
    metric_id
    operator
    expected_value
    severity
  }
  class MetricResult {
    metric_id
    state
    value
    evidence
  }
  class AssessmentFinding {
    requirement
    result
    state
  }
  class Certificate {
    confirmed_context
    findings
    metrics
    quality_profile
    reproducibility
  }
  Model <|-- Goal
  Model <|-- DatasetContext
  Model <|-- Certificate
  DatasetContext *-- Goal : contains
  Goal o-- Requirement : requirements list
  AssessmentFinding *-- Requirement : declared criterion
  AssessmentFinding *-- MetricResult : measured evidence
  Certificate *-- DatasetContext : frozen context
  Certificate o-- AssessmentFinding : findings list
  Certificate o-- MetricResult : diagnostics list
''','Triangles show inheritance. Diamonds show fields that contain other validated objects; they are not SQL relations. JSON serialization copies these structures into saved records. The common Model base is Pydantic, not the detector network.',['backend/context.py'])

add('dataset-models','Dataset and persistent record are different object types','class',r'''
classDiagram
  class Dataset {
    name : str
    sha256 : str
    modality : str
    frame : DataFrame
    profile : dict
    objects : list
    disk_path : str or None
    close()
  }
  class DataProfile {
    row_count : int
    missing_count : dict
    duplicate_rows : int
    to_dict()
  }
  class Record {
    id : String primary key
    kind : String
    created_at : String
    document : Text
  }
  class Store {
    root : Path
    engine
    create(kind, payload)
    get(id, kind)
    update(id, changes)
    list(kind)
  }
  Dataset --> DataProfile : profile derived from data
  Store --> Record : SQLAlchemy persistence
  Dataset ..> Record : persist serializes selected fields
''','Dataset and DataProfile are dataclasses. Record is the sole SQLAlchemy table model. Store stores a JSON document, not a pandas object or pickled Dataset. The conversion arrow is serialization, not inheritance.',['backend/ingestion.py','metrics/dataProfiler.py','backend/workbench/datasets.py','sql/store.py'])

add('object-lifecycle','The lifetime of a confirmed goal','flowchart',r'''
flowchart TD
  A["Browser message string"] -->|"MessageRequest JSON"| B["project_goal"]
  B -->|"interpret returns typed object"| C["DatasetContext confirmed=false"]
  C -->|"model_dump JSON"| D[("context record: draft")]
  D -->|"review response"| E["GoalChat draft state"]
  E -->|"ConfirmRequest"| F["confirm_project validates"]
  F -->|"new ID, confirmed=true"| G[("context record: confirmed")]
  G -->|"workflow pointer"| H["assess_project model_validate"]
  H -->|"ReadinessRequest -> job config"| I["worker.run reconstructs DatasetContext"]
  I -->|"assess"| J["Certificate.confirmed_context"]
  J -->|"immutable saved evidence"| K["training provenance / report / analytics"]
''','The goal moves through Python validation objects, JSON records and React state. A new interpretation clears current workflow result pointers, but historical context records and jobs remain. Editing is blocked while current compute is active.',['backend/context.py','backend/workbench/workflow_api.py','backend/workbench/worker.py','backend/readiness_engine.py'])

add('split-safeguards','Prepare partitions before model training','flowchart',r'''
flowchart TD
  A["create_split: eligible detection dataset"] --> B["load full image hashes + optional groups"]
  B --> C["union exact duplicates, near images, recording groups"]
  C --> D{"Complete declared partitions?"}
  D -->|"yes"| E["preserve train/val/test; reject overlapping clusters"]
  D -->|"no declarations"| F["seeded whole-cluster allocation"]
  E --> G["nonempty train/val/test + checksum"]
  F --> G
  G --> S[("split record")]
  S -->|"training subset only"| R["readiness assessment"]
  S -->|"development vs final test"| I["assert_independent"]
''','A file named test is not automatically independent. Exact hashes, perceptual signatures and comparable recording-group identifiers are checked. Missing comparable groups require acknowledgement; acknowledgement does not prove independence.',['backend/workbench/splits.py','backend/workbench/workflow_api.py','backend/workbench/detector.py'])

add('training-lifecycle','Persistent job states and recovery','state',r'''
stateDiagram-v2
  [*] --> queued : JobManager.enqueue creates record
  queued --> running : JobManager._run launches worker
  queued --> cancelled : cancel before start
  running --> completed : worker.run saves result
  running --> failed : worker catches exception or exits unexpectedly
  running --> cancelled : manager terminates owning process
  running --> interrupted : recover after backend restart
  interrupted --> queued : new resume job if training checkpoint valid
  cancelled --> queued : new resume job if training checkpoint valid
  failed --> queued : new resume job if training checkpoint valid
  completed --> [*]
  note right of queued: Resume creates another job ID; it does not rewrite history
''','The resumed arrow is a new training record, not mutation of the interrupted run back to queued. Experiments also retain configured/queued/running/completed/failed/cancelled/interrupted status; the parent owns inline child jobs.',['backend/workbench/jobs.py','backend/workbench/worker.py','backend/workbench/lab_api.py','backend/workbench/detector.py'])

add('training-functions','The actual detector training call chain','flowchart',r'''
flowchart TD
  A["api.training / execute_experiment"] -->|"TrainRequest config"| B["worker.run task=training"]
  B --> C["detector.train"]
  C --> D["validate_training + device"]
  D --> E["materialize -> write_yaml"]
  E --> F["setup_detector + fixed target initialization"]
  F --> G["YOLO.train: actual PyTorch training"]
  G -->|"on_fit_epoch_end callback"| P[("job progress + run/results.csv")]
  G -->|"run/weights/best.pt"| H["finish_training"]
  H -->|"validation + hashes + manifests"| M[("model record")]
  M -->|"model_id"| R["completed job result"]
''','Training performs real library computation. YAML and cached image/label paths describe train and validation inputs; final test images are excluded. Epoch callbacks report actual completion, loss, validation and GPU memory, not fabricated percentages.',['backend/workbench/detector.py','backend/workbench/worker.py','backend/workbench/experiments.py'])

add('experiment-data-flow','A guided controlled comparison becomes a named experiment','sequence',r'''
sequenceDiagram
  participant UI as GuidedValidation
  participant WF as workflow_api
  participant API as api job/record reads
  participant Lab as lab_api.start_experiment
  participant Q as JobManager
  participant Worker as worker / execute_experiment
  participant DB as Store
  UI->>WF: validation/prepare (intervention, severity)
  WF->>DB: variation + split + workflow_validation
  WF->>Q: enqueue changed-data readiness
  UI->>API: poll jobs / workflow-validations
  API->>DB: read saved records
  DB-->>API: current state
  API-->>UI: JSON state
  UI->>WF: validation/protocol after readiness completes
  WF->>DB: create_experiment freezes validated protocol
  UI->>Lab: POST experiments/{id}/start
  Lab->>Q: enqueue experiment parent
  Q->>Worker: launch parent experiment process
  Worker->>Worker: child training for each mode and seed
  Worker->>Worker: final evaluation after all training
  Worker->>DB: models, evaluations, experiment outcome
  UI->>API: poll completed records
  API->>DB: read final records
  DB-->>API: outcomes
  API-->>UI: JSON models / evaluations
''','The browser reads records through HTTP handlers and never opens SQLite. Inside a parent worker, child run calls are serial and share the worker process; there is no subprocess per child.',['frontend/src/GuidedValidation.tsx','backend/workbench/workflow_api.py','backend/workbench/experiments.py','backend/workbench/jobs.py'])

add('model-evaluation','Two evaluation passes produce complementary metrics','flowchart',r'''
flowchart TD
  A["evaluate_models: model IDs + final-test config"] --> B["test_manifest + classes + assert_independent"]
  B --> C["materialize canonical test boxes"]
  C --> D["YOLO.val: confidence sweep for AP"]
  D -->|"mAP, per-class AP, curves"| R["evaluation metrics"]
  C --> E["YOLO.predict: operating confidence per image"]
  E --> F["detection_errors: class-aware matching"]
  E --> G["confusion_counts: spatial matching ignoring class"]
  F -->|"TP / FP / FN; precision / recall / F1"| R
  G -->|"predicted rows / true columns"| R
  R --> S[("evaluation record + predictions.json")]
  S --> U["BenchmarkCharts / side-by-side predictions"]
''','Average precision comes from Ultralytics; operating-point errors come from repository matching code. The confusion matrix deliberately uses a separate matching rule to expose wrong-class detections. They must not be treated as the same calculation.',['backend/workbench/detector.py','backend/research/experiments.py','frontend/src/BenchmarkCharts.tsx','frontend/src/GuidedValidation.tsx'])

add('research-analytics','Join results, control comparability, then analyze','flowchart',r'''
flowchart TD
  E[("evaluation records")] --> R["analytics.research"]
  M[("model + readiness + dataset records")] --> R
  R --> C["configuration/context/policy/test signature"]
  C --> A["aggregate repeated seeds within image source"]
  A --> D["exclude shared images/groups, artificial, combined, imported"]
  D --> P["Pearson / Spearman on independent comparable sources"]
  D --> L["predictive_comparison: at least 6 sources"]
  R --> V["paired intervention outcomes on related data"]
  P --> O["Research DTO: rows, aggregates, correlations, limitations"]
  L --> O
  V --> O
  O --> UI["Analytics charts / scoped experiment report"]
''','Repeated seeds are not new datasets. Related variants can support paired observations but are excluded from independent-dataset correlations. Missing or unmatched evidence often leaves charts empty for valid reasons.',['backend/workbench/analytics.py','backend/workbench/api.py','backend/workbench/reporting.py'])

add('reporting-flow','Reports reuse stored evidence rather than ask a model','flowchart',r'''
flowchart TD
  U["Export link in report / experiment UI"] --> A["workbench.api.report(id, format)"]
  A --> D{"Record and requested format"}
  D -->|"readiness JSON / CSV"| J["saved record / flattened metric rows"]
  D -->|"readiness HTML"| H["reports.html_report"]
  D -->|"PDF or research HTML"| R["reporting.document"]
  R -->|"saved certificate or selected-model research"| P["pdf_document / html_document"]
  J --> O["downloadable response"]
  H --> O
  P --> O
  O --> B["browser download / readable report"]
''','The application report generator is deterministic and reads saved evidence. Its own PDF may abbreviate long structured values; the visual guide exports created for this task retain the entire guide instead of using that application report template.',['backend/workbench/api.py','backend/reports.py','backend/workbench/reporting.py'])

add('storage-architecture','Where the application remembers things','flowchart',r'''
flowchart TD
  U["Browser"] -->|"IDs, theme, selection"| L["localStorage / sessionStorage"]
  U -->|"HTTP"| A["FastAPI process"]
  A -->|"legacy only"| M["memory: datasets / interpretations / jobs"]
  A -->|"workbench"| S[("metadata.sqlite3: research_records")]
  S -->|"JSON paths/references"| D["datasets/id: frame + evidence + table"]
  S -->|"job IDs"| J["jobs/id: log + config + YAML + checkpoints"]
  S -->|"cached source paths"| C["cache / imagecache / variations"]
  S -->|"prediction path"| E["evaluation JSON on disk"]
  A -->|"temporary intake"| T["owned staging / DuckDB spill directories"]
''','Only one application table exists; references inside JSON are application-managed, not foreign-key columns. Archive/delete semantics differ between transient legacy sessions and durable research records.',['main.py','sql/store.py','backend/workbench/datasets.py','backend/workbench/detector.py','backend/large_ingestion.py'])

add('error-recovery','Failure paths preserve uncertainty and useful evidence','flowchart',r'''
flowchart TD
  A["User operation"] --> D{"Failure location?"}
  D -->|"input / lookup"| V["ErrorRoute: 422 or 404"]
  D -->|"remote context invalid"| C["interpret: identified local fallback"]
  D -->|"metric evidence absent"| M["run_metric: insufficient_evidence"]
  D -->|"compute exception"| W["worker.run: failed + worker.log"]
  D -->|"backend stopped"| R["JobManager.recover: interrupted"]
  V --> U["frontend shows error; correct request"]
  C --> H["human reviews and confirms"]
  M --> Q["unknown score / requirement; supply evidence"]
  W --> N["new run / checkpoint resume if supported"]
  R --> N
''','No branch turns an error into invented accuracy. Cancellation, provider fallback and metric uncertainty have different meanings. Out-of-memory errors get a specific smaller-batch/image suggestion; checkpoint resume requires a valid retained training checkpoint.',['backend/workbench/api.py','backend/context_extractor.py','metrics/core.py','backend/workbench/worker.py','backend/workbench/jobs.py','backend/workbench/lab_api.py'])

from current_diagrams import refresh
refresh(DIAGRAMS, add)
