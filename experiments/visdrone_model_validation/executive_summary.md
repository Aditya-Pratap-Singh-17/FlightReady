# Executive summary

Seven controlled YOLO11n runs were completed on real VisDrone data. Clean achieved mAP50-95 0.0333 on the untouched official labeled test-dev split.
Every completed defect condition reduced mAP50-95. The largest loss was incomplete_40: 0.0281, down 15.6%. Wrong classes fell 12.0%, Gaussian noise 3.6%, missing annotations 1.8%, and combined defects 15.2%.
The repaired control independently reproduced clean performance exactly (0.033275), showing full observed recovery when original bytes and labels were restored.

FlightReady marked every dataset NOT_READY at 80% evidence coverage, but its numeric score was not directionally aligned with mAP. Combined defects scored 94.043 versus clean 93.013 even as mAP50-95 fell 15.2%.
Across all seven rows, Pearson r=-0.377 (p=0.405) and Spearman rho=-0.561 (p=0.190); the small one-seed matrix is descriptive, not confirmatory.

Conclusion: real dataset defects materially harmed fixed-budget fine-tuning, repair restored performance, and the present aggregate readiness score should not be treated as a calibrated proxy for downstream detector accuracy. Add semantic-label review, expected-coverage evidence, corruption checks, and transparent score contributions before making such a claim.

See research_report.md, results/performance_comparison.csv, and plots/ for the complete evidence and limitations.
