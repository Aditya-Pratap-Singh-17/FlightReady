# Final Manuscript Verification

## Status

**Professor-review draft; not represented as submission-ready.** The manuscript's scientific claims, equations, run accounting, tables, and plotted results are internally consistent with the retained FlightReady artifacts. One conference-policy issue remains unresolved: the official IEEE IV 2027 author instructions prohibit LLM-generated manuscripts and allow only light LLM editing for grammar and spelling. This draft received substantive Codex assistance. A disclosure alone does not cure that local prohibition; the researcher and professor must determine whether a fully human-authored rewrite and institutional review are required.

## Official-format audit

- Assumed track: regular/contributed paper.
- Official expectation: six pages; up to eight pages with charges for pages seven and eight.
- Initial-review format: anonymous, with author identities, affiliations, and acknowledgments removed.
- Generated PDF: 5 pages, US Letter (612 x 792 points), PDF 1.4, unencrypted, searchable text.
- Fonts: Times New Roman regular and bold are subset-embedded; no used unembedded font was found.
- Layout: two columns, numbered Roman-numeral sections, abstract, index terms, corrected displayed equation, algorithm, four tables, five figures, and numbered references.
- Visual inspection: all five rendered pages were inspected. No clipping, overlap, missing caption, broken table, orphaned final reference, or unreadable equation was observed.
- The five-page length is below the stated six-page expectation but within the eight-page maximum; the professor should decide whether additional scholarly context is required rather than adding cosmetic padding.

The editable DOCX was regenerated and structurally reopened successfully, but the packaged DOCX visual renderer could not run because LibreOffice is not installed. The PDF—not the DOCX—received the complete rendered-page visual audit. The LaTeX source uses `IEEEtran`; no TeX engine is installed in this environment, so it should be compiled in the official IEEE/Overleaf environment and checked again before any submission.

## Authorship and anonymity

- Verified researcher identity: Aditya Pratap Singh.
- Verified role: undergraduate researcher.
- Verified institution for a future camera-ready identity block: The Pennsylvania State University.
- Department, email, ORCID, publication credentials, and any coauthors remain unknown and were not invented.
- The initial-review PDF and DOCX are anonymous. The LaTeX source contains the verified camera-ready identity only in comments, with clear activation instructions.
- No professor or other person was assigned authorship.

## Independent numerical recalculation

The read-only verifier parsed the saved per-run JSON files and scorecard metric rows rather than trusting the manuscript tables. It confirmed:

| Check | Independently recalculated result |
|---|---|
| Amended plan | 19/19 planned tags complete |
| Total retained observations | 21 |
| Supplemental observations | YOLO11n Gaussian-25, seeds 42 and 93 |
| Clean YOLO11n | n=3; mean 0.4789807; SD 0.0059199; 95% t interval [0.4642749, 0.4936866] |
| Combined-40 YOLO11n | n=3; mean 0.4451552; SD 0.0044310; interval [0.4341479, 0.4561625]; paired relative change -7.0473% |
| Combined paired difference | -0.0338255; 95% interval [-0.0574140, -0.0102371] |
| Gaussian-25 YOLO11n | n=3 (1 planned + 2 supplemental); mean 0.4747040; SD 0.0023926; interval [0.4687605, 0.4806475] |
| Gaussian-25 paired difference | -0.0042767; 95% interval [-0.0229358, 0.0143824] |
| Pearson association | r=0.767204, p=0.026290, n=8 |
| Spearman association | rho=0.838338, p=0.009323, n=8 |
| Excluding repaired | Pearson r=0.739871, p=0.057287; Spearman rho=0.857143, p=0.013697; n=7 |
| Reference-free evidence | primary score withheld for every condition; coverage 0.555556 |
| Controlled-defect detection | 1/10 reference-free; 10/10 deterministic 10% audit; 10/10 full oracle |
| Conventional baselines | 5/10 |
| Fixed-update coverage control | 0.4472348; -5.3224% relative to matching clean seed |
| YOLOv8n transfer | clean 0.4860734; combined 0.4454443; -8.3586% |

Every stored condition-summary value, t interval, paired change, correlation coefficient and p-value, readiness score, coverage value, categorical decision, sensitivity count, and conventional-baseline result matched the independent recalculation.

## Equation and decision-logic audit

- Weighted geometric aggregation is normalized by total measured weight \(W_M\).
- The measured-zero case is handled directly.
- Evidence coverage uses measured applicable weight divided by all applicable weight.
- Primary quality is withheld unless coverage equals one.
- Score bounds preserve unavailable applicable evidence.
- The fixed gate order is reported consistently with the implementation.

## Claim audit

- The paper retains the positive n=8 association and its nominal p-values.
- The apparent significance contradiction is resolved: the full eight-condition tests are nominally significant at alpha 0.05, while Pearson is not significant after removing repaired (p=0.0573).
- The correlation is described only as a related-condition, descriptive association—not predictive validation, calibration, causation, independent dataset-level generalization, or safety certification.
- The historical tabular and VisDrone v2 reassessments are labeled method-development evidence.
- The 19 planned runs and two supplemental observations are distinguished in the abstract, methods, results, table note, figure caption, and verification record.
- Negative findings and limitations are retained, including poor reference-free sensitivity, the unreplicated 10% audit, single-seed conditions, split-group leakage risk, lack of an external wrong-class experiment, and engineering-selected weights/thresholds.

## Citation audit

The bibliography was checked against primary/official records for the Data Readiness Report (arXiv:2010.07213), the JDIQ and Information and Software Technology surveys (DOIs `10.1145/3592616` and `10.1016/j.infsof.2023.107268`), Data Cascades (DOI `10.1145/3411764.3445518`), Datasheets for Datasets (DOI `10.1145/3458723`), VisDrone, the UAVVaste article (DOI `10.3390/rs13050965`), the UAVVaste Zenodo record (DOI `10.5281/zenodo.8214061`), and official Ultralytics documentation. The previously incorrect authorship for the machine-learning dataset-quality survey was corrected to Y. Gong, G. Liu, Y. Xue, R. Li, and L. Meng.

## Conference and AI-policy issue requiring action

The official IV 2027 author page states: regular papers are expected to be six pages; initial submission is anonymous; acknowledgments must be removed; and manuscripts generated by LLMs are prohibited, with only light grammar/spelling editing allowed. IEEE's general conference policy separately requires disclosure of AI-generated content in acknowledgments. The stricter conference rule controls this submission unless the conference provides written clarification.

The editable source includes a proposed camera-ready disclosure identifying substantive OpenAI Codex assistance with software development, automated experiment orchestration, analysis scripting, and manuscript preparation, while assigning scientific responsibility to the researcher. That disclosure is intentionally commented out in the anonymous initial-review source. It should not be activated—or assumed sufficient—without the professor's policy review.

Official pages checked:

- https://ieee-iv.org/2027/contributions/information-for-authors/
- https://ieee-iv.org/2027/contributions/call-for-papers/
- https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
- https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/

## Final required human actions

1. Resolve the IV 2027 LLM prohibition with the professor and applicable institutional guidance; perform a fully human-authored rewrite if required.
2. Confirm track, final author list, and camera-ready affiliation/contact details.
3. Compile the LaTeX source with the official IEEE template and inspect the resulting PDF.
4. Run the conference PDF checker/PDF eXpress if required.
5. Obtain scientific and authorship approval before submission.

No new experiment was conducted, no frozen-v2 file was modified, and nothing was committed, pushed, or submitted.
