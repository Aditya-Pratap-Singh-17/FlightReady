# Local platform verification — November 1, 2026

## Executed checks

- Backend suite: **221 passed** (the original 198 tests plus 23 workbench tests).
- Frontend suite: **17 passed**, including the original 11 regressions and six dashboard tests.
- Production build: TypeScript strict checking and Vite build passed.
- Dependency consistency: `python -m pip check` reports no broken requirements.
- Microsoft Edge / Playwright: **2 passed**, exercising actual backend requests. The second test performed real training from the UI, independent evaluation, prediction comparison, comparison auditing and navigation through backend/research/experiment/settings pages. Desktop and mobile layouts were checked; page-error collection was empty.
- Real CPU smoke from scratch: candidate and reference YOLO11n models trained for one epoch, then evaluated on the same four independent labeled synthetic images. Initialization tensor hashes matched and were verified in the actual trainer. Evidence: `.runtime/smoke-30a9d8c5/verification.json`, with its SQLite records, checkpoints, configuration, logs and JSON/CSV exports.
- Real CPU fine-tuning smoke: both models initialized from the same trusted local detector checkpoint, trained and tested independently. Matched-condition comparison passed. Evidence: `.runtime/smoke-b20800d9/verification.json`. This workspace also supplied models/test data for the full browser training/evaluation check.

The synthetic fixtures are randomized 64×64 drawings with explicit labels and recording IDs. One training epoch is an execution check: measured mAP was **0**, recall **0**, FP **0** and FN **4** on the four test images; operating precision was **unavailable** because no predictions passed the confidence threshold. These are actual outputs, not substituted demo accuracy. They provide no evidence that the readiness method predicts real-world UAV performance.

## Critical covered invariants

Persistent import/reopen, existing YOLO/COCO parsing, declared absent classes, original-metric integration, training-only assessment scope, safe archive members, directory registration without import copies, immutable parsed labels, source image mutation detection, deterministic grouping/splits, exact and near-duplicate overlap checks, class mappings, thresholded detection matching, zero-denominator metrics, queue subprocess lifecycle, cancellation/interrupted state, meaningful OOM errors, schema validation, report exports, seed aggregation and controlled variations retaining validation/test bytes.

The browser check found an import-selection race: a newly imported dataset could be replaced by the previously cached library selection before refetch. The implementation and a dedicated regression test now preserve the imported selection. The detector adapter also verifies its exact effective initialization so framework model reconstruction cannot silently invalidate a matched experiment.

## Remaining verification limits

- NVIDIA driver detection succeeded for the RTX 4070 Laptop GPU, but installed PyTorch is **2.15.0+cpu**. CUDA training, GPU memory behavior and GPU determinism remain unverified. Use the official CUDA-compatible PyTorch installer and run the GPU smoke before claiming GPU validation.
- No real UAV training dataset or independent field-test dataset was supplied. No empirical readiness/performance relationship or causal conclusion has been established.
- YOLO11n scratch and local checkpoint fine-tuning ran on CPU. YOLO11s, combined-model optimization and controlled-variant optimization use the same adapter but were not separately trained in the verification run; controlled-data construction and combined-source leakage safeguards have automated coverage.
- Existing test tooling emits a Starlette/httpx deprecation warning. Vite reports ignored dependency `use client` directives during its successful client build. Neither is a test/build failure.
- Video/tracking, segmentation, telemetry detector training, multiclass confusion-matrix plotting, distributed execution, PDF export and automatic checkpoint resume remain unsupported as documented in `research-platform.md`.

Validation artifacts live under ignored `.runtime` directories; no datasets/checkpoints/logs were added to Git. No repository commit was created, and existing user files were retained.
