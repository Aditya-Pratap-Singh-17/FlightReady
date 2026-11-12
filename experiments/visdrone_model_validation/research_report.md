# Real VisDrone and YOLO11n validation

## Completed measured results

All listed results use official VisDrone2019-DET test-dev images and labels. Training used the same downloaded YOLO11n initialization, 320 px resolution, five epochs, batch 16, seed 42, and CUDA GPU unless the row states otherwise.

| Variant | Degradation | FlightReady score | Coverage | Decision | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | reference | 93.01305825992891 | 80.0 | NOT_READY | 0.0735 | 0.0333 | 0.2672 | 0.1094 |
| combined_40 | combined | 94.0426518312471 | 80.0 | NOT_READY | 0.0616 | 0.0282 | 0.1447 | 0.0998 |
| combined_repaired | oracle_repair | 93.01305825992891 | 80.0 | NOT_READY | 0.0735 | 0.0333 | 0.2672 | 0.1094 |
| gaussian_noise_25 | image_noise | 93.01498995518467 | 80.0 | NOT_READY | 0.0710 | 0.0321 | 0.2600 | 0.1055 |
| incomplete_40 | dataset_completeness | 93.157674478496 | 80.0 | NOT_READY | 0.0619 | 0.0281 | 0.3408 | 0.0987 |
| missing_annotations_25 | missing_annotations | 93.64665430381703 | 80.0 | NOT_READY | 0.0715 | 0.0327 | 0.2689 | 0.1056 |
| wrong_class_25 | wrong_class | 93.01305825992891 | 80.0 | NOT_READY | 0.0643 | 0.0293 | 0.2624 | 0.0998 |

## Findings and hypothesis outcomes

1. **Data degradation harmed the fixed-budget detector (supported in this run):** all 5 completed defect conditions reduced test mAP50-95 versus clean (True). Relative losses ranged from -1.8% to -15.6%.
2. **Semantic and coverage failures were material:** wrong classes reduced mAP50-95 by 12.0%, and removing 40% of training images reduced it by 15.6%.
3. **Combined defects compounded selected individual defects (partially supported):** combined corruption reduced mAP50-95 by 15.2%, worse than Gaussian noise (3.6%) and missing annotations (1.8%), but not lower than the incomplete condition (15.6%).
4. **Oracle repair restored observed performance (supported for this seeded run):** repaired mAP50-95 was 0.033275, exactly matching clean to reported precision; its readiness score also matched clean.
5. **FlightReady score did not predict detector performance in this matrix (not supported):** scores increased for several harmful variants. Correlations were negative and not statistically significant. This is evidence of metric misalignment/blind spots for these mechanisms, not proof that readiness is never predictive.

## FlightReady diagnostic implications

- Gaussian corruption changed the readiness score by only +0.002 points while reducing mAP50-95 by 3.6%.
- Structurally valid wrong-class labels received the exact clean score while mAP50-95 fell 12.0%; semantic correctness needs gold labels, cross-model disagreement, or targeted human review.
- Missing annotations and combined defects received *higher* scores than clean (93.647 and 94.043 vs 93.013). Annotation-density reductions can make internal statistics look cleaner while weakening supervision.
- Corpus completeness cannot be inferred from retained records alone. A manifest-level expected count, scene/route coverage targets, or acquisition-plan evidence is needed.
- Recommended product changes: expose metric-level score contributions; add annotation-density/coverage baselines; accept external gold-review samples; add image-corruption checks; and never present the aggregate score as a validated proxy for downstream mAP until prospective calibration succeeds.

## Statistical relationship

{
  "n": 7,
  "pearson": {
    "r": -0.37670938057181375,
    "p": 0.40487258366593903
  },
  "spearman": {
    "rho": -0.5609681940050744,
    "p": 0.19014684492175052
  },
  "reason": null,
  "defects_only": {
    "n": 5,
    "pearson": {
      "r": -0.13725602747556342,
      "p": 0.8257904821954042
    },
    "spearman": {
      "rho": -0.09999999999999999,
      "p": 0.8728885715695383
    }
  }
}

The all-row correlation includes clean and the byte-identical repaired control and is descriptive only. The defects-only analysis removes those duplicated endpoints. Neither analysis is powered for inference (n=7 and n=5).

## Reproducibility and provenance

- Dataset: official VisDrone2019-DET train (6,471 images), val (548), and labeled test-dev (1,610); official ignored regions were excluded during YOLO conversion.
- Model: official downloaded YOLO11n checkpoint; SHA-256 and archive hashes are recorded in provenance.json and each result JSON.
- Protocol: five epochs, 320 px, batch 16, seed 42, deterministic mode, official val for checkpoint selection, official test-dev once for final reporting.
- Full variant manifests record deterministic affected filenames, severity, and prevalence. Unchanged files are hard-linked; official downloads remain preserved.

## Sources

- VisDrone official repository: https://github.com/VisDrone/VisDrone-Dataset
- Ultralytics VisDrone dataset documentation: https://docs.ultralytics.com/datasets/detect/visdrone
- Ultralytics YOLO11 model documentation: https://docs.ultralytics.com/models/yolo11

## Scope and limitations

- The official archives and downloaded checkpoint are preserved under resources/downloads with hashes in provenance.json.
- The VisDrone official repository/catalog supplies no explicit dataset license; redistribution rights are not inferred.
- Five-epoch, 320 px runs are controlled resource-bounded fine-tuning experiments, not state-of-the-art VisDrone training.
- One training seed does not estimate training-run uncertainty. Test-dev was used only for final evaluation, never model selection.
- Oracle repair restores known original bytes/labels; it is not evidence for an automatic real-world repair algorithm.
- Any planned variants without a result JSON were not executed and are not presented as successful.
