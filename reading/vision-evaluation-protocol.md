# Independent validation procedure for estimated UAV properties

The repository's fake-model tests verify software plumbing only. No real annotated
estimation benchmark or pretrained-model accuracy result is claimed.

1. Sample held-out flights/sessions/sources independently of encoder/classifier
   development and controlled training variants. Pre-register target classes,
   operational weather/lighting/scene/occlusion definitions, sampling strata,
   metadata trust rules, abstention policy and annotation protocol.
2. Have independent annotators label condition categories, uncertain/unobservable
   cases, actual object appearance distinctions and ambiguous cross-class pairs.
   Use collection instruments/records where visual appearance cannot establish
   weather, capture time, altitude or calibrated viewpoint. Record reviewer IDs,
   disagreements, adjudication and external evidence. Do not use model predictions
   to create these labels. Keep this audit immutable and content-addressed.
3. Freeze model name/version/hash, preprocessing, label mapping and inference
   configuration before examining evaluation labels. Run the local adapter on the
   held-out inputs. Keep all class predictions and uncertainty summaries, including
   failures/abstentions; no selectively discarded difficult samples.
4. For condition classifiers, report the category confusion matrix, class-wise
   precision/recall, macro metrics, missing/abstained coverage, calibration/reliability
   curves and pre-registered error summaries. Stratify by flight/sensor/deployment
   conditions. Compute uncertainty intervals using a justified flight/session-level
   resampling design rather than treating adjacent frames as independent.
5. For embedding-based diversity and confusing-pair candidates, use independently
   reviewed pair relevance and target-class identity. Evaluate reviewed candidate
   precision and retrieval recall on a separately labeled candidate population.
   Compare encoder/configuration sensitivity and background confounds. Embedding
   dispersion is a model-dependent feature, not directly a correctness probability.
6. For label-preserving transformations, independently review retained object
   visibility, box geometry, semantics and sensor plausibility. Only transformations
   meeting declared assumptions may be described as invariance evidence; record
   rejected/uncertain transformations too.
7. Validate relationships to downstream performance using the separate experiment
   framework: matched budgets/sizes when feasible, several seeds, independent
   validation/test groups, immutable evaluation annotations, condition-specific
   detector results and collateral-property measurements. Do not claim metric
   importance, causal effects or accuracy improvement before these experiments.

Required external fixture layout for a future audit: immutable image content
hashes, flight/session split IDs, independent reviewer category labels and
uncertainty flags, annotation guidelines/version, adjudications, reference records,
and fixed model configuration. Real independently annotated fixtures have **not**
been fabricated or bundled; supplying and reviewing them is a research prerequisite.
