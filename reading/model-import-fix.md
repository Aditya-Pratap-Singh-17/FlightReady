# Model Lab import repair - 10 November 2026

The previous implementation exposed only a fixed YOLOS registry entry, hid import
behind readiness/goal checks, did not reliably refresh imported models after a
download, and reused old training-eligibility flags for inference. Those paths
have been repaired. The running app was restarted on its existing port, 5423,
using the same `.runtime/workbench` dataset/project store.

## Use the repaired lab

1. Refresh FlightReady and open **Model Validation Lab**. Import is available
   before goal confirmation or readiness analysis.
2. **Hugging Face**: paste a model ID or its page URL, optionally change the
   revision, and choose **Inspect Hugging Face model**. The app resolves the
   exact commit and displays architecture, labels, license and download size.
   Review/approve the weights download, then choose **Download & import model**.
3. Alternatively select **Local model folder** and enter a Transformers
   `save_pretrained` directory containing config, processor config and safetensors
   weights. The app copies the safe artifact into its managed library.
4. Choose the dataset. Imported models refresh and select automatically.
5. **Preview predictions** runs without matching labels or a readiness report;
   it makes no accuracy claims. The default preview covers ten sorted images;
   its recorded limit can be changed.
6. **Benchmark selected annotation classes** computes actual COCO-style AP and
   operating-point precision/recall for the explicitly selected compatible
   classes. Unscored classes and unvalidated project requirements are disclosed.

Supported built-in Transformers detector families are **YOLOS, DETR and RT-DETR**.
Repositories are not restricted to a fixed catalog; architecture support is.
LLMs, custom remote Python architectures, pickle-only weights, private/gated Hub
authentication and other model tasks are not implemented by this importer. Use
a local safe export for a private supported detector.

Examples inspected successfully against the live Hub: `hustvl/yolos-tiny`,
`facebook/detr-resnet-50`, and `PekingU/rtdetr_r18vd`. Only YOLOS Tiny weights were
downloaded, following explicit user consent. Metadata inspection does not fetch
other models' weights or transmit dataset contents.

Built-in concrete model/processor classes load locally with safetensors and
remote code disabled. DETR/RT-DETR do not silently acquire extra backbone weights.
Sharded safetensors are supported with validated index references. Import records
retain the immutable revision or local file hashes. The optional inference
requirements now include `timm` for DETR backbones.

## Actual verification on the user's saved dataset

YOLOS Tiny is present in the real model library as model ID
`325b5810aa5b46e8b059f450bd3cfe90`, from `hustvl/yolos-tiny` at revision
`95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de`.

- A real prediction preview ran on three images from the saved 548-image
  VisDrone2019-DET validation import; it produced 76, 71 and 80 predictions at
  confidence 0.25 and image size 320. No quality scores were fabricated for that
  preview.
- A real CPU benchmark then evaluated **all 548 images**, restricted explicitly
  to bicycle, car, truck and bus. mAP@0.50:0.95 was **0.040267** and mAP@0.50 was
  **0.086263** (fractions). Operating confidence was 0.25, matching IoU 0.50,
  preprocessing size 320, and the adapter retained up to 100 predictions/image.
- The scored classes contained 16,352 ground-truth objects: 4,780 matched,
  11,572 missed; there were 16,271 unmatched predictions. These measurements
  apply only to the recorded class scope and protocol.
- Benchmark ID: `3c86281cc24b4e55b50928bb898a7f91`. Its predictions, configuration,
  dependency versions, source/model hashes and exports persist in the user's
  existing store. The lab lists it when that VisDrone dataset is selected.

These are ordinary canonical-box evaluations, not the official VisDrone
ignore-region protocol. The excluded classes are pedestrian, people, van,
tricycle, awning-tricycle and motor. This ground-object benchmark **does not**
validate detecting drone targets or nighttime operation. A pretrained COCO
detector lacks a drone-target label; previews remain useful, but cannot establish
that capability.

## Software verification

- Full backend run during this repair: **348 passed**.
- Final focused backend regression run: **55 passed**, including real local
  inference with tiny YOLOS, DETR and RT-DETR, legacy eligibility flags, repository
  inspection/acquisition, local imports, scoped metrics and unscored previews.
- Frontend: **37 passed**; production TypeScript/Vite build passed.
- New browser journey passed with an actual pretrained local model import,
  automatic selection, preview and annotation benchmarking without readiness.
  Desktop/mobile screenshots were rendered and inspected; no horizontal
  overflow was found.
- The existing guided paired-training browser journey also passed after the
  import changes, including two actual training runs and held-out evaluation.
- The repaired live app's runtime, custom Hugging Face inspection endpoint and
  persisted model library were checked after restarting it.

Evidence: `.runtime/pretrained-user-verification.json`,
`.runtime/pretrained-visdrone-benchmark.json`, `.runtime/hf-inspection-verification.json`,
`.runtime/hf-import-final-backend.log`, `.runtime/hf-import-final-frontend.log`,
`.runtime/model-import-browser.log`, and `.runtime/model-import-verification/`.

Primary API references:
[Hub model information](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/package_reference/hf_api#huggingface_hub.HfApi.model_info),
[Transformers RT-DETR](https://huggingface.co/docs/transformers/v4.57.1/model_doc/rt_detr).
