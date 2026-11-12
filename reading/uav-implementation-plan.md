# UAV extension: repository audit and plan

The existing application uses `metrics.REGISTRY` / `CALCULATIONS`, strict Pydantic
contracts, `readiness_engine.assess`, and a two-worker assessment executor. Archive
members are inspected without extraction. Image summaries and objects are bounded
by the current archive/image/object limits; disk tables expose only a preview.

Existing coverage: UAV-01/14 brightness and contrast; UAV-08 normalized boxes;
UAV-10/11 native VisDrone flags; UAV-15 box/class syntax, annotation presence,
COCO/VOC manifest checks and independent tabular labels; UAV-17 reference JS/W1
comparisons; UAV-18 exact/perceptual duplicates and split/group overlap; UAV-19
declared sensor fields. Reuse these rather than replace them.

Incremental implementation:

1. Extend contracts additively with measurement status, provenance, units and
   scope; register UAV-01..20 in the existing registry.
2. Add deterministic pixel, object, metadata, temporal and sensor summaries.
   Preserve absence versus zero and document denominators and bounded sampling.
3. Add explicit contextual bins/reference distributions and optional local visual
   evidence; never download models or trust metadata as independent ground truth.
4. Derive six learning views from the same results plus conditional nuisance
   associations; integrate JSON/HTML/UI and confirmed requirement comparison.
5. Test known synthetic cases and the existing API/format/security regressions.
6. Add independent, opt-in training-variant and experiment export utilities.

Compatibility risks: legacy `state=pass` describes successful calculations;
new UAV diagnostics use `state=warning` and separate `assessment_status` until a
requirement is evaluated. Disk previews must never masquerade as full-data UAV
measurements. Optional visual computations must execute only in the existing job
worker. Retained image sources must follow dataset lifecycle cleanup. No existing
research inputs may be overwritten.

Baseline (before edits): 104 passed, 49 failed, 1 error. Failures are missing
PyYAML in `.venv` and denied default pytest temporary-directory access. The
project-local `.packages` and an explicitly writable pytest temp directory will
be checked before attributing any failures to implementation changes.

Completion: the declared PyYAML dependency was installed into `.venv`; the
workspace pytest temp/cache paths resolved the access error. Baseline then passed
154 tests. All eight implementation phases are represented in code and documented
in `uav-framework.md`; final regression verification is recorded in
`verification.md` (198 backend tests, 11 frontend tests, production build passed).
Optional model adapters and detector orchestration are tested with unit-test
doubles. Real independent annotation/model validation and downstream experiments
remain research work, not fabricated completion evidence.
