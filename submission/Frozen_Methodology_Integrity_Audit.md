# Frozen Methodology Integrity Audit

## Audit conclusion

The source identity of FlightReady v2 is verified. I found no evidence that the frozen metric definitions, normalizations, thresholds, weights, aggregation, coverage rule, or decision logic changed after the freeze. Three manifest hashes match their current files exactly. The fourth manifest value is an invalid 63-character SHA-256 string caused by one omitted hexadecimal character; the corrected 64-character value matches the current CRLF file exactly, and the LF-normalized file is byte-for-byte identical to the research-release Git blob.

I preserved `experiments/independent_validation/frozen_methodology.json` unchanged. This audit documents an archival transcription defect; it does not redefine or modify the frozen methodology.

## Frozen record

- Methodology version: `flightready-v2.0`
- Freeze timestamp: `2026-11-12T12:00:00-05:00`
- Recorded base commit: `510d24eead326d1565dd29ddaf19ac059fcc6347`
- Research-release commit containing the four audited blobs and manifest: `29758973836c22ea06c88dec0e4f472306045a5e`
- Audit-time repository HEAD: `8d964c36f70ce4905366311f0d53e2554e647100`
- The four frozen source paths are Git-clean relative to audit-time HEAD.

## File-by-file verification

| Frozen file | Manifest value | Current raw SHA-256 | LF-normalized SHA-256 | Comparison with research-release blob | Status |
|---|---|---|---|---|---|
| `metrics/readiness_v2.py` | `7fc1864711e2e9a7b14a2e8e08aacb54c590e4d9ab2f84ec50185c821e4d54fc` | same | same | 7,651 bytes; byte-identical | Exact match |
| `backend/readiness_engine.py` | `f24fb6520329822fc7a9345e381bbcfeddbbf048763ad8f023c2d58edc8a28f` (63 characters) | `f24fb6520329822fc7a9345e381bbcfeddebbf048763ad8f023c2d58edc8a28f` | `2fbcda6f0e7dc7dac63d48e4ad8e74fdf5576a9a7785db48fb39603115242900` | 19,902 LF-normalized bytes; byte-identical | Manifest transcription defect resolved; source verified |
| `metrics/config/adaptive_quality.json` | `e0f00149fbd11d1b907a58320780276150d1ba60608e2be25be1d961a041f7e5` | same | `1fce95d0f810e056fb7a53714511b3501746a595b5aa7e9b2510a6e24caea765` | 23,733 LF-normalized bytes; byte-identical | Exact raw match; line-ending normalization verified |
| `metrics/quality_aggregation.py` | `280848a97a18e991ceeddacc122221af70f8b9533e210fd56b44933b280179b2` | same | `21c8d7514f52ae34bb6380cd127f2342eec2f250c7e791d97fcb2ac9a136ade5` | 21,763 LF-normalized bytes; byte-identical | Exact raw match; line-ending normalization verified |

The `backend/readiness_engine.py` manifest value omits the `e` that appears after `...bcfedd` in the exact raw hash. Its 63-character length is impossible for a complete SHA-256 digest. The current file contains 251 CRLF line endings; normalizing them to LF produces the exact blob stored in commit `2975897`. The other two CRLF files show the same expected raw-versus-normalized behavior, but their manifest entries recorded the raw Windows hashes correctly.

## Mathematical implementation check

For applicable metrics \(A\), measured metrics \(M\), positive weights \(w_i\), and normalized measured scores \(q_i\in[0,100]\), the implementation uses

\[
W_A=\sum_{i\in A}w_i,\qquad W_M=\sum_{i\in M}w_i,\qquad c=\frac{W_M}{W_A},
\]

\[
Q_M=100\exp\left(\frac{1}{W_M}\sum_{i\in M}w_i\ln\frac{q_i}{100}\right).
\]

The denominator is the sum of measured weights, not a metric count. A measured zero with positive weight returns zero directly. Not-applicable metrics are excluded; unavailable applicable metrics remain in the coverage denominator. The primary quality score is available only when coverage is exactly 1.0. Lower and upper bounds assign unavailable applicable metrics zero and 100, respectively.

The categorical decision order is also verified from source: assessment error; measured blocking failure; unknown blocking evidence or coverage below 0.8; measured warning; otherwise ready. A partial numeric summary cannot override these gates.

## Evidence linkage and limitations

The retained scorecards identify methodology version `flightready-v2.0`; the experiment records, configurations, protocol amendments, and analysis outputs consistently refer to the frozen method. The saved experimental JSON files do not independently embed all four full source hashes, so the Git history, manifest, and file-byte comparisons jointly establish identity. The invalid manifest digest should be disclosed with this audit rather than silently corrected in the frozen record.

## Non-actions

- No experiment was run for this audit.
- No frozen methodology file or manifest was edited.
- No commit, push, or submission was performed.
