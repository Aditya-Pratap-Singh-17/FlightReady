# Experimental summary

The experiment completed on a real 2,580-row dataset with a fixed clean test set, 18 train-data conditions, two model architectures, and three seeds (108 fitted models).

When training quality changed from the cleaned reference to the combined 40% low-quality condition, FlightReady's experimental weighted score changed from 100.00 to 100.00. Hist Gradient Boosting R² changed from 0.1635 to 0.0573 (-0.1062); RMSE changed from 0.9557 to 1.0145 (+0.0588). Oracle repair restored R² to 0.1635 and the score to 100.00.
When training quality changed from the cleaned reference to the combined 40% low-quality condition, FlightReady's experimental weighted score changed from 100.00 to 100.00. Random Forest R² changed from 0.1567 to 0.0720 (-0.0847); RMSE changed from 0.9595 to 1.0066 (+0.0470). Oracle repair restored R² to 0.1567 and the score to 100.00.

The numeric pipeline score is experimental, not a validated overall readiness percentage. Its coverage and categorical readiness decision are included with every result. The main scientific caveat is that schema-valid label and feature corruption can harm models without an equivalent score penalty.

See `data_quality_impact_report.md` for all variants, correlations, plots, limitations, and the evidence-based assessment.
