# Dataset formats and import choices

Upload local data downloaded from Kaggle, GitHub or another source. The application reads files; it does not download URLs, authenticate to Kaggle or run repository code. A successful import establishes readable structure, not correct labels or fitness for a mission.

## Workflow

1. Upload a file or supported archive. Open **Parsing options** for an unusual delimiter, missing header, format override, JSON/XML record path or document-level text records.
2. If the archive, JSON object or workbook contains multiple candidates, choose one and click **Load selected dataset**. The staged upload is reused. Train and test data are never silently concatenated.
3. Inspect the detected schema, labels and parsing notes. **Use detected dataset setup** updates editable goal fields and requires confirmation again. Confirm the task, target and requirements yourself.
4. Run assessment. The completed panel provides results and downloadable HTML/JSON reports, including failed and insufficient-evidence decisions.

## Text and tables

| Input | Interpretation |
|---|---|
| CSV, TSV, TAB, PSV, delimited DATA/DAT | Comma/tab/semicolon/pipe detection; explicit delimiter overrides guessing. Set `has_header: false` for generated `column_1`, etc., or provide `column_names`. Ragged rows and duplicate headers are rejected. |
| TXT, TEXT, MD, RST, LOG | One line per record in `text`; blank lines remain missing observations. Consistent tab-separated TXT is recognized as a table. Comma prose remains text unless explicitly configured. `text_unit: "document"` reads one document. |
| JSON | Array of records, one object, or records in a wrapper array. Ambiguous arrays require `record_path`, for example `data.items`. Nested values are canonical JSON strings, not discarded. |
| JSONL, NDJSON | One JSON object/value per nonblank line, with schema union across records. Invalid lines and duplicate object keys produce errors. |
| XML | Repeated record elements, inferred when unambiguous; otherwise select slash-separated `record_path`, for example `dataset/record`. Nested values are preserved. DTDs/entities are rejected. |
| XLSX | Read-only worksheet records. Choose `sheet_name` for multiple populated sheets; formulas remain text and are never executed. Legacy XLS is unsupported. |
| Parquet | Batched scalar tables. Nested/binary Parquet columns remain unsupported. |
| Feather, Arrow IPC | Record batches; nested values are serialized as JSON fields. |
| Archive of text documents | Select **Text corpus** for one document per file with `text`, `label`, `source_file`. Folder labels are inferred organization and must be verified. |

JSON string identifiers such as `"001"` remain strings. Boolean and numeric source types are retained for type checks. New fields appearing late in a record stream become columns, with earlier values missing. Structured JSON fields cannot establish a scalar type requirement. Plain text import provides structural checks; semantic correctness, language quality, sentiment labels and model suitability are not automatically established.

## Images

| Layout | Behavior |
|---|---|
| Single static image or unlabelled collection | Decodes pixels and records dimensions/hashes; does not invent labels. |
| `train/cat/a.jpg`, `train/dog/b.jpg` (ImageFolder) | Classification labels from class directories; common train/val/test directories become split metadata. Confirm folder meaning. A single class or partial folder layout may require choosing classification versus unlabelled interpretation. |
| YOLO `images/...`, `labels/...` | Normalized `class_id x_center y_center width height`. Names from classes.txt, .names or safe YAML `names`. Without a mapping, `class_id:N` remains explicit and semantic class coverage is insufficient evidence. |
| VisDrone DET `images/...`, `annotations/...` | Native eight-field detection ground truth, with ignored regions and non-target categories retained separately. See README. |
| COCO JSON plus images | Original category/image IDs; pixel xywh boxes validated against actual decoded dimensions. Crowd flags are preserved. Orphan references, duplicate annotation IDs and missing files are reported. Explicit empty annotations are negatives. Segmentation masks are not evaluated. |
| Pascal VOC `JPEGImages/...`, `Annotations/*.xml` | Default coordinates are 1-based inclusive: `[1,1,width,height]` covers the full image. Set `voc_coordinate_base: 0` only for an export using zero-based continuous coordinates. Difficult/truncated flags are preserved. |

JPEG, PNG, WebP, BMP, TIFF, GIF, JFIF, PNM and AVIF extensions are recognized; the installed Pillow codec must decode the actual content. Animated/multi-frame images require conversion to explicitly selected static frames. Missing annotations are unknown, distinct from declared negative images. Box syntax cannot prove the absence of missed objects or correct human labeling.

## Containers and metadata

ZIP, TAR, TAR.GZ/TGZ, TAR.BZ2/TBZ2 and TAR.XZ/TXZ are inspected without extracting paths. GZ/BZ2/XZ files can wrap a single supported file. Links, unsafe paths, duplicate members, excessive expansion and excessive nesting are rejected. README/license/code files do not become dataset rows.

Advanced upload metadata supports `format`, `delimiter`, `has_header`, `column_names`, `csv_encoding`, `text_unit`, `record_path`, `sheet_name`, `dataset_member`, `collection_type`, `annotation_format`, `annotation_file`, `image_root`, `class_file`, and `voc_coordinate_base`. Use the UI's candidate choices for exact archive paths. For example:

```json
{"dataset_member":"repository/data/train.csv","delimiter":";","has_header":true}
```

Selection API: `POST /api/datasets/{id}/select` with `{"options": {...}}`. A pending selection cannot start an assessment. Invalid choices leave the staged upload available for correction.

## Resource limits and unsupported inputs

The default upload ceiling is 2 GiB. Text tables use disk-backed bounded batches and full-dataset metrics; the displayed preview is only 100 rows. Defaults include 512 columns, 8 MiB per text record/document, JSON/XML nesting depth 64, XLSX upload 256 MiB, archive expansion 4 GiB, 50,000 archive entries, 20 MiB/20 megapixels per image, and one million target objects. At most three archive container levels are accepted. Image rows and object metadata still occupy RAM. These are safety/working-set limits, not a promise that every 2 GiB input layout fits memory.

Malformed data produces an actionable import error rather than silently skipping damaged records. Custom binary formats, pickle/joblib, HDF5, TFRecord, PDFs/OCR, audio, video, executable dataset scripts, encrypted archives, RAR/7z, and semantic segmentation evaluation need conversion or a dedicated reader. The pipeline cannot truthfully support every arbitrary format without errors.

Readers include explanatory comments in `backend/record_formats.py`, `backend/dataset_archives.py`, `backend/universal_images.py`, and `backend/large_ingestion.py`. Regression examples live in `tests/test_dataset_formats.py`.
