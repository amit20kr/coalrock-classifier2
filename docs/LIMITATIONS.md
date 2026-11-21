# LIMITATIONS.md

> This file is part of the SpectraSense evaluation evidence trail.
> Every limitation here corresponds to a question an interviewer may ask.

## 1. Data provenance and split methodology

The dataset contains **10,497 unique spectra** (after removing 23 exact duplicates from 10,520 raw rows)
across 24 lithology classes — 12 coal (fuel) and 12 rock (gangue) types — collected at the
[Chinese University of Mining and Technology and affiliated mines]. The split is **row-level stratified
random** (seed 42): train 7,353 / val 1,578 / test 1,566.

**What this split does not prove:**
- Spectra from the same physical specimen may appear in both train and test sets. A 1-NN cosine
  similarity leakage probe shows that a fraction of test spectra have near-identical training
  neighbours; see `artifacts/baselines.json` for the exact figure.
- There is no leave-one-mine-out, leave-one-instrument-out, or leave-one-measurement-condition-out
  validation. Reported accuracy is within-distribution accuracy on held-out rows, not
  out-of-domain generalization.
- The 1,566 test spectra represent laboratory conditions: controlled illumination, static samples,
  clean sensor surface, optimal measurement distance. Real conveyor-belt operation involves
  vibration, dust, moisture, varying granularity, and mixed-material scenes — none of which
  are represented in the data.

## 2. Measurement conditions not represented

The model has **never been tested on**:
- Moving belt or vibrating samples
- Wet or dust-coated surfaces
- Mixed-material scenes (e.g., coal and gangue touching in the same measurement spot)
- Different sensor hardware or wavelength calibrations
- Mines not included in the training dataset

Do not deploy this system as the sole decision-maker in a physical environment without
first conducting an in-situ validation study with known reference samples.

## 3. Safety net thresholds

The gangue override threshold (0.95 total gangue probability mass per scan) and minimum flagged scan
count (k = 4) were **chosen on validation data only** via expected-cost minimisation with an assumed
miss-to-false-alarm cost ratio of 10:1. This ratio is an engineering assumption, not a measurement
of actual business costs. Any deployment must recalibrate thresholds using site-specific cost data.

Pure-coal batches can false-reject when enough scans have high gangue probability. Per-class performance is recorded in artifacts/metrics.json; batch simulation results are in artifacts/batch_metrics.json.

## 4. Confidence calibration

The confidence shown in the UI is the **mean-pooled top-1 softmax** over the entire batch. Softmax
outputs are not calibrated probabilities. Temperature scaling on the validation set is not yet applied
to the deployed model. Treat the confidence figure as a relative ranking, not a probability.

The REVIEW band triggers at confidence < 0.60 (a heuristic, not a calibrated threshold).

## 5. Noise robustness

One noise test from the original submission injected Gaussian noise into the **SNV channel** after
preprocessing, not into the raw reflectance domain. This overstates robustness because SNV removes
albedo and distance variation before the noise is added. The figure "76.6% accuracy at 26 dB SNR"
refers to SNV-channel perturbation. Robustness to raw-domain noise has not been evaluated.

## 6. Model architecture claim

The model is a **Pre-Convolutional ViT (PCViT)**, adapted from Yang et al., *Energies*
2022, using the CLS token for classification. It is a single-scale model.
The training code was adapted from a reference implementation; the architecture was not designed
from scratch.

## 7. Baselines

The comparison is single-seed (no mean ± std). McNemar's test has not been run. The baseline
comparison is provided as indicative evidence that the model exceeds naive methods, not as a
statistically rigorous superiority claim.

## 8. Input format: the model is validated only on the raw dataset layout (measured 2026-11-11)

The training notebook's loader (`surgical_load`) keeps the 5 numeric metadata rows of the raw dataset
files (light angle, distances, azimuth) as the **first 5 "bands"** of every spectrum and front-crops to
1500, so every training and test spectrum is `[5 metadata values, bands 1000..2494 nm]`. The serving
pipeline reproduces this exactly for that layout (reported as `dataset_raw`, `validated: true`).

A **clean** 1500-band spectrum (metadata stripped, bands 1000..2499 nm) is a different input to the model.
Measured with the deployed checkpoint on the two real raw files available for this review:

| File (true class) | Raw dataset layout | Clean 1500-band layout |
|---|---|---|
| Carbonaceous shale (rock), 418 scans | 99.8% exact class; 0 scans called coal | 68.9% exact class; **116 of 418 scans called coal** (rock→coal) |
| Coking coal (fuel), 418 scans | 22 of 418 scans called rock only under the clean layout; exact class 3.1% | |

Adding a fixed 5-value prefix restored most of the rock result (92–98%) but not the coal class labels, so
no cheap adapter fixes this. Consequences:
- The API labels every response with `input.format` and `input.validated`; the UI shows an `UNVALIDATED` tag.
- Unvalidated-input REVIEW is enabled. Missing or malformed settings resolve to enabled, and an explicit disabled setting refuses startup. Unvalidated input changes an otherwise ACCEPT result to REVIEW. REJECT retains precedence.
- **Proper fix (needs GPU/Colab, not done):** rebuild the training arrays without the metadata prefix,
  retrain, and re-run `ml.evaluate`. Until then, "95.72% accuracy" applies to the raw dataset layout only.
- Two files, one rock class and one coal class, is a small sample. Re-run `tests/test_real_demo_file.py`-style
  checks for every class before generalising.

## 9. What the k = 4 rule does and does not catch (measured, real data)

Injecting genuine rock scans (Carbonaceous shale) into a 300-scan real coal batch (Coking coal):

| Rock scans injected | Verdict |
|---|---|
| 0 | ACCEPT |
| 1, 2, 3 | ACCEPT + warning ("inspect these scans manually") |
| 4, 5, 10, 50, 200 | REJECT (Safety Override) |

The large-file rule is unchanged. In an otherwise ACCEPT batch above the cutoff, 1–3 flagged
scans among hundreds produce a warning and the batch is still auto-accepted. The existing
batch-mean and four-flag REJECT rules still take precedence.

For windows of `review_any_flagged_up_to_scans` scans or fewer, one or more scans at or
above the same 0.95 gangue flag threshold changes an otherwise ACCEPT verdict to REVIEW. The
warning gives the flagged count and threshold. The inclusive cutoff defaults to 20. With the
same single flagged scan and all other ACCEPT conditions met, a window of 20 scans goes to REVIEW,
while a window of 21 scans remains ACCEPT with a warning.

The cutoff must be at least the maximum scans per lump the sensor delivers. Twenty is a lab
replay default. Revisit it when the sensor and belt speed are known. If scans per lump can exceed
20, raise the setting and test the new boundary before field use.

The independent-window replay estimates an additional pure-coal REVIEW rate of +0.3 percentage
points at 3 scans, +1.3 at 10 and +2.3 at 20. The estimates use held-out scans from the same
specimens as training and assume independent scans. Repeated scans from one real lump may be
correlated. Measure the REVIEW rate in a pilot. Field conditions remain untested, including
distance variation beyond the lab's fixed 0.5 m setup, moisture, dust, and new seams. See
[the lab replay](FEW_SCAN_EVALUATION.md) and [its tables](../artifacts/few_scan/). The replay uses
R0 for the previous decision rule and R1 for the small-window change described here.

## 10. Spectral-feature claims are unverified

On the two real files tested, a scale-free prominence measure of the 2nd-derivative at 2150–2250 nm did
**not** separate coal from carbonaceous shale (median 3.2× vs 3.6× the noise floor), and the 1900 nm window
was indistinguishable. Carbonaceous shale is carbon-rich and may be a weak clay example, so this is not
evidence against the physics, but it means the dashboard must not claim to *detect* Al-OH or H₂O. The
UI shows shaded literature windows labelled "not detections". Verify with clay/kaolinite classes and an
attribution method before any such claim returns.

## 11. Resource limits

On a 5.4 MB raw file (418 scans) the backend peaked at about 430–500 MB RSS in testing (chunk size 8; it was
640 MB at chunk size 64). This is tight for a 512 MB instance; use 1 GB for demos with large files.
Uploads are capped at 10 MB and 2000 scans; larger files are rejected, never subsampled.

## Current held-out result

95.72% accuracy on 1,566 held-out test scans, with coal-to-coal / rock-to-rock / coal-to-rock / rock-to-coal counts of 741 / 820 / 4 / 1. One rock scan was called coal. Source — artifacts/metrics.json, multi_class.accuracy and error_decomposition.

Split and preprocessing sources — notebooks/coalrock_preprocess2.ipynb. Noise and CPU latency measurements — notebooks/coalrock_training_evaluation.ipynb. Safety thresholds — backend/assets/safety_config.json and artifacts/batch_metrics.json. Earlier serving measurements above describe their original runs and are not field validation.

Observed over all 24 raw CSVs, the retained 1500-value model input and its 1495 retained reflectance bands range from -5.32500982 to 102.855812 percent-equivalent values; the five header entries are metadata, and the earlier 0–100 reflectance bounds excluded observed data. Source — artifacts/reflectance_range.json.

The observed retained reflectance range is −5.325 to 102.856 percent (artifacts/reflectance_range.json). Validation now accepts −20 to 120 percent. This extra margin is a design choice, rather than a measured physical or sensor specification. All scans in the 24-file corpus lie inside it. Per-class counts and band positions outside 0–100 are in artifacts/reflectance_outliers.json.
