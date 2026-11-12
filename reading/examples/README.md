# Computed synthetic report

These JSON/HTML artifacts were computed from the repository's existing
`demo_data/detection_valid.zip`, without modifying it. They demonstrate the actual
engine and are not real UAV benchmark results, independently reviewed annotations,
or evidence of detector accuracy. `READY` applies only to the example's confirmed
measurable requirements; unavailable UAV properties remain explicitly unavailable.

Reproduce into a new directory:

```powershell
.\.venv\Scripts\python.exe backend/scripts/generate_uav_example.py --output .runtime/my-uav-example
```

The existing demo archive SHA256 is
`d3439ddf83ad03af7b9828cd12da84123acf28ff2c9c2c315b9bcd993ca0e859`.
Dependency versions and the context are recorded in the generated report.
