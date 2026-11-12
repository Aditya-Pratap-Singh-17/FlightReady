"""Current, source-reviewed narrative revisions for the November 3 pipeline.

Exports still read only the assembled Markdown and its rendered diagrams.
The detailed metric specification is incorporated into the book, not merely
linked as an external addendum.
"""
from pathlib import Path
import re

DOC = Path(__file__).resolve().parents[1]
SNAPSHOT = '3 November 2026'


def specification_sections():
    text = (DOC/'research-metrics-v2.md').read_text(encoding='utf-8')
    headings = list(re.finditer(r'^## (.+)$', text, flags=re.M))
    return {m.group(1): text[m.start():headings[i+1].start() if i+1<len(headings) else len(text)].strip()
            for i,m in enumerate(headings)}


def update_manuscript(text, roles, glossary, debug):
    sections = specification_sections()
    roles.update({
        'metrics/uav_metrics/research.py':'Versioned geometry, quantiles/bins, explicit preprocessing and shared three-valued image/object scenario intersections.',
        'metrics/uav_metrics/pixels.py':'New cached global pixel features and one bounded, deterministic object/ring inspection shared within each assessment session.',
        'metrics/uav_metrics/audit.py':'Independent-reference one-to-one annotation audit categories, image/object denominators and explicit negative/candidate evidence.',
        'metrics/uav_metrics/independence.py':'Exact hashes, declared collection groups and resource-capped dHash/BK-tree near-duplicate candidate components.',
        'metrics/uav_metrics/presentation.py':'Eight chart/table evidence views derived from MetricResult values; it makes no separate readiness decision.',
        'frontend/src/ResearchEvidence.tsx':'Shared guided/advanced evidence selector, actual count/distribution charts, unavailable states and provenance/denominator tables.',
        'frontend/src/researchEvidence.test.tsx':'Synthetic UI checks for real zero versus unavailable evidence, source metadata and historical certificates.',
        'frontend/e2e/research-evidence.spec.ts':'Actual browser upload/goal/readiness/evidence/mobile smoke test that checks no models were trained.',
    })
    text['ch01'] += r'''
## The current pipeline, in one continuous journey

This edition incorporates engine **1.5.0**, enhanced UAV property implementation **2.0**, evidence schema **research-evidence-2.0** and **interventions-2.0**. The public twenty UAV IDs and six independent quality dimensions remain stable. The following is the implemented path, including the newer research evidence.

1. Upload/import stages a managed source, validates its structure, parses canonical labels and inspects pixels. The image frame now caches the existing dHash signature and new global degradation indicators alongside historical measurements.
2. The goal assistant proposes a DatasetContext. A separate confirmation saves the finish line, including any explicit scale bins, detector preprocessing, joint scenarios or numerical minimums. Interpretation never independently validates collection metadata.
3. Training-scope analysis selects only training images. Image-level audits/candidates are scoped to those images without editing parent records. An old count-only audit cannot be assumed to describe the selected subset.
4. The existing registry executes requirements, diagnostics and stable UAV properties. New modules provide geometry, bounded local crops/rings, independent-reference audits, negative coverage, direct scenario conjunctions and collection/duplicate evidence. They do not create a parallel assessment engine.
5. Requirement findings determine Certificate.overall_status. Stored observations also feed the existing six-dimension policy with strict coverage and independent gates. New uncalibrated proxies add no automatic score weights or blockers.
6. A presentation adapter derives eight evidence views from actual MetricResult values. The certificate persists both raw evidence and its readable views. Guided and advanced interfaces use the same ResearchEvidence component; exports preserve missing values and recorded scope.
7. The Model Validation Lab can create reproducible training-only interventions, reassess changed data, freeze a matched protocol, run explicitly requested real training and evaluate on independent final data. Native-size recall is now recorded separately from standard AP. Research relationships remain unconfirmed until sufficient real outcomes exist.

Think of the new modules as additional instruments attached to the same laboratory bench. An instrument may return a useful observation while its policy score remains unavailable. More instruments do not themselves establish a better model or an empirically calibrated readiness score.
'''
    text['ch04'] += r'''
## How the newer evidence reaches a chart

`readiness_engine._assess` passes serialized metric results to `uav_metrics.presentation.evidence_sections`. Its result is stored as `Certificate.quality_measurements.research_evidence`. `QualityAnalytics` mounts `ResearchEvidence`; the guided workflow reaches that same component through its **Quality & visual evidence** tab. There is one selected evidence view at a time rather than a separate assessment implementation in the browser.

Eight views cover object scale, local complexity, object clarity/contrast, whole-image degradation, independent audits, target negatives, conditional scenarios and collection independence. A selector chooses the quantity within a view. Each plot identifies its units/denominator, observed evidence type, sample size, coverage and provenance. Tables expose actual backend counts, including real zeros. Unsupported distributions show unavailable instead of an invented empty zero chart. The source/method disclosure retains sampling and calculation versions. Historical certificates remain readable and explicitly say when enhanced evidence was never collected.

Changing a card or plot reads stored evidence; it does not invoke another image scan, a language model or a training run. A newly requested assessment is the operation that recollects evidence. Configuring a scoring preview remains a distinct operation.
'''
    text['ch06'] = text['ch06'].replace('ownership and source path.', 'ownership, source path and an optional runtime-only `image_reader` capability.')
    text['ch06'] += r'''
## Cached global pixels, local reads and correctly scoped audits

`inspect_image` stores a hexadecimal `dhash64` derived from the same 9x8 grayscale comparison used by the split service. It also calls `pixel_features` once on the existing <=256-pixel proxy to cache protected `research_*` global fields. Brightness, contrast, entropy, historical Laplacian variance and bytes/pixel keep their earlier semantics. Metadata merging cannot replace these computed fields.

`datasets.load` supplies an in-process reader built from `image_source`. It can read verified managed archives, registered directories or virtual combined sources. The capability is never serialized into evidence.json and never supplied as uploaded executable code. `subset` preserves the reader but filters the frame, objects and hashes to selected images. A local object inspection can therefore obtain only its chosen source members. It checks image hashes/dimensions and decode budgets again, then shares one bounded result across UAV-12/14 within the assessment session.

`subset` deep-copies metadata. A structured independent audit bound to the parent (or not yet bound) is filtered to selected image names, receives the subset fingerprint and records the parent audit/source fingerprint. Whole-population count summaries are removed from that derived audit. An unscoped count-only audit is unavailable for a new subset instead of being extrapolated. Candidate hard-negative image records are similarly filtered. The parent dataset and historical certificate are unchanged. Combined readiness keeps original source audits in provenance and does not promote them into a union-wide audit; local pixel readers can still inspect the union's namespaced sources.

Older frame caches may lack the new global fields or dHash. Their absence is unavailable evidence; the code does not fill it with zero or decode an unbounded archive simply to manufacture a complete result. Reimporting a source collects the new ingestion fields. Reassessment can collect available bounded local evidence under its recorded configuration.
'''
    text['ch08'] = text['ch08'].replace(
        'UAV estimated output paths retain their estimated status.',
        'UAV estimated output paths retain their estimated status. Projection refuses unavailable parent/submeasurements and unverified semantic status leaves; an inspection count of zero is not substituted for an unavailable quality measurement.')
    text['ch08'] = text['ch08'].replace(
        'Optional image model work has its own bounded session behavior.',
        'Optional image model work has its own bounded session behavior. New object-local pixel measurements reuse that same assessment-local session cache, so UAV-12 and UAV-14 share their image decodes. After cached ingestion distributions, the engine also stores evidence_sections from the actual metric results before adaptive aggregation.')
    text['ch08'] = text['ch08'].replace(
        '`scenario_count` intersects all declared conditions and optional class presence. Separate daytime and urban counts do not establish that any urban-daytime example exists. If the matching count is zero and relevant metadata is unknown, the check cannot conclude true absence. The current sample-sufficiency metric deliberately asks for a confirmed minimum instead of deriving one universal sample count from a formula.',
        '`scenario_count` delegates to the same three-valued scenario_evidence used by UAV-17. Separate daytime and urban counts do not establish any urban-daytime example. Unknown metadata/annotations are kept separate from known nonmatches; false dominates unknown in a conjunction. Object predicates must refer to the same object. A positive observed count is a lower bound when unknown candidates remain: an already-met minimum can pass, while an otherwise unmet minimum remains insufficient evidence. Named scale conditions require explicit confirmed object_size_bins. No universal sample minimum is inferred.')
    text['ch08'] += '\n## Enhanced research evidence: complete current definitions\n\n'
    text['ch08'] += 'The following specification is incorporated into this textbook from the maintained metric document, after checking the current implementation. These are additional submeasurements under stable IDs, not a replacement for the execution order or decision models above.\n\n'
    for heading in ('Measurement map','UAV-08: scale geometry','UAV-12 / UAV-14: global and object-local pixels','UAV-15: independent audit','UAV-17 and shared conditional scenarios','UAV-18: collection groups and duplicate candidates'):
        text['ch08'] += sections[heading]+'\n\n'
    text['ch09'] += r'''
## Why the v2 observations do not silently change the scoring model

The six-dimension `adaptive-1.0` definition, engineering weights, normalization and independent gates are retained. A ring-entropy value, contrast ratio or collection-group count does not have a universally justified quality direction. Consequently the additional raw submeasurements are inspectable without automatically entering more weighted components or creating blockers. Explicit confirmed requirements may project measured fields; they continue through existing findings and gates.

The historical annotation-agreement component consumes the older independently reviewed error fraction. The structured audit has separate reference-object, annotation and image denominators; the code does not collapse those categories into that older fraction. A new scored audit component would need an explicit future policy version, scientific rationale and validation. Count-only audits without image scope cannot be reused as a new training-subset audit. Old saved results are not rewritten.

Changing tasks invalidates the new context-dependent UAV evidence as well as older target/requirement evidence. A preview cannot relabel a sampled object analysis, class-specific negative count or a scale scenario as evidence for another task. Historical and new assessments keep their own fingerprints, including engine/metric implementation versions; preserving the policy definition does not mean new measurement fingerprints should equal older ones.
'''
    text['ch10'] = text['ch10'].replace('with blur, resolution reduction, label omission or box perturbation;', 'with blur, resolution reduction, label omission, box perturbation, wrong-class labels, pixel noise or underexposure;')
    text['ch10'] += r'''
## Expanded training-only interventions and native-size outcomes

The guided adapter maps wrong-class changes to existing annotation_noise and maps noise/exposure to image_degradation. Noise is a recorded seeded achromatic Gaussian perturbation broadcast across RGB, clipped/rounded to 8-bit; exposure is a recorded brightness factor. Existing omission and box-center perturbations remain. The manifest records parameters/seed, parent/output hashes, parent/output object counts, removed training images and Python/NumPy/Pillow/Pandas versions. These transformations do not alter independent validation/test image or label bytes.

Advanced variants can also reduce whole training images containing objects below an explicit native-area criterion, retain declared classes or environmental conditions, and reduce explicitly recorded target-free hard-negative candidates. Small-object reduction uses whole-image selection, not selective deletion of small-object labels. Hard-negative reduction requires a complete explicit boolean candidate column and target-free annotations; it does not establish that those images cause false positives. These selections can change image count, class/scene mixture and other quality properties together. Count-matched controls and measured covariates are essential for interpreting an experiment.

`native_size_outcomes` reuses the actual operating-point matching results. For each ground-truth native-area bin, TP equals its labelled-object population minus its missed-object count; recall is TP divided by that population, or null for an empty bin. False positives have no ground-truth size, so their counts are separately binned by predicted native box area. Evaluations store `recall_small_native`, `recall_medium_native` and `recall_large_native`, plus size-stratified counts and a size-evaluation protocol version. The default descriptive area bins are [0,1024), [1024,9216) and [9216,infinity).

Small-object AP remains explicitly unavailable. Native-box operating recall is not official COCO mask-area AP and cannot be used to reconstruct an AP confidence sweep. The research outcome selector exposes these real recall values and false positives per image alongside existing outcomes. No real training/evaluation is automatically triggered by adding a measurement or opening its view.
'''
    text['ch11'] = text['ch11'].replace('`numeric_metrics` extracts finite nonboolean numeric leaves from measured diagnostics and available dimension scores.', '`numeric_metrics` extracts finite nonboolean numeric leaves from measured diagnostics and available dimension scores. It now skips unavailable nested evidence and sampling/settings/protocol/criteria/bin-definition constants; an unavailable local count is not a numerical zero predictor.')
    text['ch11'] += r'''
## Gates, new outcomes and an explicit pending research state

The research-only predictor `baseline.critical_gates_pass` is 0 when a gate conclusively fails and 1 when all gates conclusively pass; unresolved gate evidence does not receive an invented value. It supports investigating gate compliance alongside existing equal/adaptive arithmetic/geometric baselines while leaving the six product dimensions independent. The implementation does not presume that adaptive weighting or gating wins an empirical comparison.

Recorded native-size recall and false positives/image can be selected as actual outcomes. Individual v2 measurements can join the existing comparable-source analysis only when they are finite and available. Related variants remain paired evidence and do not inflate independent dataset counts. New measurement, scenario and evaluation protocol identities remain part of interpreting comparability; unmatched settings or source overlap must not be ignored merely to populate a plot.

The API returns empirical_validation.status=pending when no actual outcomes exist, and the research UI explicitly says empirical validation is pending. Recorded outcomes change that to outcomes-available, not to scientifically-validated. A useful next study independently audits annotation subsets, freezes preprocessing/scenarios and final tests, applies matched interventions with repeated seeds, then replicates across independent UAV sources. No test-fold performance may be used to tune the reported quality-policy weights.
'''
    text['ch12'] += r'''
## Where the v2 evidence lives

New global proxies and dHash are protected frame columns saved in frame.json. Structured audits and hard-negative candidate declarations enter through dataset metadata in evidence.json, with explicit source/reviewer/protocol settings; references are scoped for derived populations. Local crop/ring measurements are bounded in-memory observations during assessment, not a database of every full-resolution crop. The certificate stores distributions, class/size strata, sample sizes, configuration, limitations and bounded candidate references in its metric values.

`quality_measurements.research_evidence` stores a readable derivative of those metric results. It is not a new SQL table or an alternative engine. JSON and readiness CSV carry raw structured values; current HTML/PDF report renderers add readable evidence sections. The application PDF can abbreviate long methodology/raw structures and explicitly points to JSON; this textbook's separate exporter continues retaining the complete book. The runtime-only image_reader is reconstructed by load and cannot be serialized as uploaded code.
'''
    text['ch14'] = text['ch14'].replace('Cached pixel measurements feed `collect_measurements`; `aggregate_quality` produces supported normalized criteria and six dimensions.', 'Cached pixel measurements feed `collect_measurements`, and evidence_sections adds eight research views from actual metrics; `aggregate_quality` produces supported normalized criteria and six dimensions.')
    text['ch14'] += '\n'+sections['Configuration examples']+'\n'
    text['ch14'] += r'''
## Following one new measurement from input to report

Suppose an explicitly configured 100x50 image has one valid box of normalized width/height 0.2/0.2. Its native box is 20x10 pixels, area 200 pixels squared and minimum side 10. Under a confirmed 640x640 letterbox with scaleup=true, gain is 6.4, giving a 128x64 input box; padding does not change those dimensions. With scaleup=false, it remains 20x10. These are illustrative geometry calculations, not measurements of a personal dataset or detectability predictions.

The image loader caches global evidence. A training subset retains only its relevant independent references. UAV-08 calls scale_analysis; UAV-12/14 share local_measurements; UAV-15 returns structured audit/negative evidence; UAV-17 and requirement scenario_count share scenario_evidence; UAV-18 returns bounded collection/duplicate evidence. Their MetricResults enter the same Certificate. presentation.evidence_sections creates chart/table views from those results, and the UI/export reads that stored certificate. A missing ring, independent audit or confirmed scale criterion stays unavailable throughout that journey.

For a new controlled noise/exposure/class intervention, the route still follows prepare -> reassess -> protocol review -> explicitly requested training -> independent evaluation. Compare the intended metric change and any collateral size/class/scenario changes before interpreting performance. With no real run, the research view remains pending. A successful synthetic workflow is software evidence, not a successful UAV experiment.
'''
    text['ch16'] += r'''
## Debugging the research-evidence branches

For missing model-input scale, inspect the confirmed detector_preprocessing shape/mode/scaleup rather than assuming square stretching. For missing named scale scenarios, require explicit object_size_bins. For missing object-local values, inspect staged source access, content hashes, tiny/border crops, ring pixel count and comparison budgets before treating null as low clarity. Missing old cache columns require reimport for ingestion-wide features.

For an unavailable independent audit, inspect evidence_type, independence declaration, reviewer/source/protocol, reference completeness, dataset binding and selected-image scope. Unverified model predictions cannot become reference truth. Syntax/presence failures are separate from missing-object claims. For surprising category rates, compare each reference/annotation/image denominator rather than summing incompatible rates.

For no duplicate candidates, inspect signature coverage, tree visits, result/link caps and complete_retrieval; an exhausted query is not proof of independence. For an empty empirical dashboard, check actual evaluation records and strict configuration/source-independence filters. Inspect metrics/uav_metrics/presentation.py and ResearchEvidence.tsx only after confirming the raw metric exists; the browser does not calculate replacement values.
'''
    text['ch17'] = text['ch17'].replace('General feature mutual information, physical sensor calibration validity, semantic label correctness, exhaustive annotation completeness and real target-model generalization require additional evidence/implementations.', 'General feature mutual information still requires a declared estimator implementation, and physical sensor calibration validity requires appropriate calibrated evidence. Semantic label correctness and exhaustive annotation completeness require independent references/review; the structured audit now compares such references when supplied. Calibrated visibility probabilities, statistical effective sample size, small-object AP and real target-model generalization remain unsupported or empirically unverified.')
    start='Earlier work in this session reported 277 backend tests, 29 frontend tests and five browser tests passing, including synthetic CPU training.'
    end='Static analysis cannot prove that every possible dynamic branch has executed; the guide names that uncertainty instead of inventing runtime coverage.'
    replacement='The completed metric implementation recorded 326 backend tests, 32 frontend tests, the production build and one readiness-only browser E2E passing. A 17-page synthetic application report was also checked. Those are recorded software validation results, not newly executed application tests in this documentation refresh and not evidence of real UAV accuracy. This book refresh separately verifies source references/hashes, full-content exports, diagrams, offline browser interaction and PDF layout. Static analysis cannot prove every dynamic branch executed; real audits, training, independent final tests and cross-source replication remain necessary.'
    a=text['ch17'].index(start);b=text['ch17'].index(end,a)+len(end)
    text['ch17']=text['ch17'][:a]+replacement+text['ch17'][b:]
    protocol=(DOC/'research-metrics-validation-protocol.md').read_text(encoding='utf-8')
    protocol=protocol[protocol.index('## Questions and preregistered hypotheses'):]
    text['ch17']+='\n## Real-data research plan and reproducibility checklist\n\nThe following plan is incorporated in full from the maintained research protocol. Its unchecked items are future scientific work, not completed experiments.\n\n'+protocol
    for key in text:
        text[key]=text[key].replace('\ufffd','"')
    glossary.extend([
        ('Background ring','Expanded box neighborhood excluding the target and known annotated boxes; not a segmentation-derived true background mask.'),
        ('Identification bounds','Range of possible coverage from known matches through matches plus unknown candidates; not a statistical confidence interval.'),
        ('Structured independent audit','Protocol-defined comparison of dataset annotations to complete declared human/trusted external references, with category-specific denominators.'),
        ('Hard-negative candidate','Recorded target-free review candidate with source/reason and model settings when relevant; not verified difficulty or a measured false positive.'),
        ('Native-size recall','Operating-point TP / labelled objects within native ground-truth box-area bins; distinct from size-specific AP.'),
        ('Bounded BK-tree retrieval','Hamming-distance candidate search with explicit visit/result/link caps; incomplete retrieval does not establish no duplicates.'),
    ])
    debug.extend([
        ('Local clarity/clutter unavailable','Check source/hash access, object sampling, crop size, ring pixels and neighbor budgets; inspect recorded sampling scope.','metrics/uav_metrics/pixels.py;metrics/uav_metrics/presentation.py;frontend/src/ResearchEvidence.tsx','tests/test_research_metrics.py;frontend/src/researchEvidence.test.tsx'),
        ('Audit disappears in a training subset','Image-level references are scoped; unscoped count-only audits cannot be extrapolated. Inspect parent audit hash and reference binding.','metrics/uav_metrics/audit.py;backend/workbench/datasets.py','tests/test_research_metrics.py'),
        ('Named small-object scenario unknown','Confirm object_size_bins explicitly; descriptive default bins do not define a task acceptance criterion.','metrics/uav_metrics/research.py;metrics/core.py;backend/readiness_engine.py','tests/test_research_metrics.py'),
        ('Near-duplicate count seems too small','Inspect signature coverage and complete_retrieval alongside tree/result/link limits. Do not substitute candidate count for independence.','metrics/uav_metrics/independence.py;backend/workbench/splits.py','tests/test_research_metrics.py'),
    ])
