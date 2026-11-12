# Visual codebase guide: current pipeline verification

Verified on 3 November 2026 against the current source snapshot. The guide now incorporates engine **1.5.0**, enhanced UAV property implementation **2.0**, **research-evidence-2.0** and **interventions-2.0**. It also incorporates the current backend/frontend/metrics/sql/reading folder layout, refreshed source links, and the extracted SQL table models and DuckDB connection module.

## Deliverables and common source

| Edition | Absolute path | Result |
| --- | --- | --- |
| Markdown | `C:\DataReadiness\reading\CODEBASE_VISUAL_GUIDE.md` | Complete maintained source |
| Standalone HTML | `C:\DataReadiness\reading\CODEBASE_VISUAL_GUIDE.html` | Complete offline interactive edition |
| PDF | `C:\DataReadiness\reading\CODEBASE_VISUAL_GUIDE.pdf` | Complete 299-page textbook |

Both exporters read the same generated Markdown and its actual rendered diagram assets. Neither independently rewrites the architectural explanations. Markdown SHA-256: `cfd501e939fd4fbbf922538f16f43c6ce653d7e9001a16d847dcf78c3bd23fb5`.

The current book contains 17 chapters, three worked examples, 61 detailed function cards, 851 Python symbols and 152 TypeScript declarations from 83 Python and 23 TypeScript files, including tests. The source snapshot has 74 repository-declared route decorators. Counts describe static source analysis, not full dynamic execution coverage.

## What changed in this edition

- The complete upload/confirmation/scoped-analysis/evidence/decision/scoring/report/training/evaluation journey now includes the v2 instrument modules.
- Chapters 6-9 explain cached global pixels, runtime source readers, bounded object/ring inspection, scale geometry/preprocessing, independent audits, negatives, three-valued joint scenarios, capped duplicate retrieval and unchanged six-dimensional scoring rules.
- Chapters 10-12 explain the expanded train-only interventions, native-size recall, actual-outcome/pending research states, comparability and current evidence lineage/storage.
- The worked examples include current configuration/audit schemas and a concrete native-to-letterbox size calculation.
- The new function cards and refreshed complete symbol reference cover the actual new files and current source line numbers.
- Existing pipeline/dispatch/evaluation/reporting/data-model diagrams were refreshed. Two new diagrams show shared research instruments and independent audit matching/scope.
- The full real-data research plan and reproducibility checklist are included inside the book, with pending work clearly separated from implemented software.

## Completeness and link checks

| Check | Actual result |
| --- | --- |
| Source text blocks in HTML | 8,013 of 8,013 |
| Source text blocks in PDF | 8,013 of 8,013 |
| Missing PDF or HTML text blocks | 0 |
| Missing source/asset references | 0 |
| Markdown internal links checked | 151 |
| Repository source line links checked | 1,177 |
| PDF outline entries/bookmarks | 109 |
| PDF internal link annotations | 216; all destinations valid |
| PDF source/asset/external link annotations | 3,309 |
| PDF diagram image pages | 25 |
| Out-of-page text/image bounds | 0 |
| Application/metrics/config/test source hashes checked | 122; unchanged during export |
| Rasterized pages | 299 of 299 |

Text accounting compares the visible Markdown token stream with extracted PDF/HTML text. 11 paragraphs interrupted by running page furniture or repeated table headers were additionally verified through a body-only extraction stream. They are present, not omitted. Source references were checked against real local files and line ranges. Source links require those repository files at their referenced paths; the exported books themselves are independently readable when copied elsewhere.

## Diagram and layout checks

All 25 diagrams were parsed/rendered by actual Mermaid 12.2.0 through installed Microsoft Edge/Playwright: 19 flowcharts, three sequences, two class diagrams and one state diagram. SVG/XML/viewBox checks passed. Each figure has exact Mermaid source, SVG and a PNG at 2.5 times the SVG dimensions. The HTML embeds all SVGs; the PDF uses actual high-resolution diagram images, not raw Mermaid code.

Large figures use dedicated A4/A3 portrait/landscape spreads. The minimum estimated label size after PDF scaling is 8.85 points. Body text is 10.2 points; source/reference tables are 8.3 points. Main content fonts are embedded, with standard PDF list markers.

All 13 current contact sheets covering all 299 pages were visually reviewed. Full-size inspection checked the new folder boundaries, SQL schema/persistence responsibility table and updated core dependency diagram. No clipped diagrams, overlapping text or missing sections were observed. Chapter starts and dedicated figure spreads intentionally allow white space.

## Actual offline Chrome interaction

Installed Google Chrome opened the standalone HTML through file:// with HTTP(S) requests blocked. It made zero page network requests, used zero external assets and produced zero JavaScript errors.

- All 17 chapter/sidebar targets and internal contents links exist.
- All 25 SVG diagrams have viewBox dimensions.
- Search returned 23 actual matches for `aggregate_quality`; selecting a result navigated to its text.
- Diagram zoom/fit controls worked with embedded SVGs.
- Expand/collapse operated on all 10 explanations; PDF explanations are expanded.
- The edition has 25 locally syntax-highlighted example blocks.
- At a 390-pixel viewport there was no horizontal document overflow; the mobile chapter menu opened, navigated and closed correctly.

Desktop, search, zoom and mobile screenshots are retained in the review directory. HTML and PDF have the same complete source text.

## Scope of the layout refresh

The application was reorganized into backend/, frontend/ and metrics/, with database models/connections/persistence in sql/ and reading material in reading/. Imports, launch paths, worker paths, static frontend serving and metric configuration paths were updated. The existing SQL table definition and runtime storage location were preserved. The demo_data directory was removed; browser fixtures now generate their own temporary samples.

The final source snapshot contains 122 checked application/configuration/test files. Their hashes remained unchanged while documentation was exported and verified. The guide retains the current pipeline explanations and refreshes the full source reference and diagrams against the new paths. Documentation tools statically inspect source and do not read environment-file values, import application entry points, train models or call paid context providers.

Application verification was also performed for this reorganization: 326 Python tests passed, 32 frontend tests passed, the frontend production build passed, and the real upload/context/readiness/export/responsive browser test passed against an isolated local store. The optional real-training browser test was skipped. Windows' existing pytest temp-directory permissions required a fresh workspace-local test directory. The browser check warmed the installed ML runtime's compute endpoint before its connection assertion. Real UAV scientific validation and repeated model-training studies remain pending.

## Evidence and regeneration

- [PDF/source/link/content verification JSON](assets/architecture/verification.json)
- [Offline Chrome verification JSON](assets/architecture/html-verification.json)
- [Book/source counts](assets/architecture/guide-manifest.json)
- [Actual Mermaid render manifest](assets/architecture/render-manifest.json)
- [Source-to-export text accounting manifest](assets/architecture/export-content-manifest.json)
- [All current contact sheets and selected full-size/browser previews](assets/architecture/review/)
- [Regeneration instructions and tooling requirements](README.md#refresh-the-book-after-code-changes)

Curated manuscript/diagram revisions and the maintained research specification/protocol must be reviewed when code behavior changes. Refreshing signatures alone cannot establish that an older architectural explanation is current.
