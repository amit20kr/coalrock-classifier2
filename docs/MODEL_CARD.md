# MODEL_CARD.md — SpectraSense PCViT

## Model Details

| Field | Value |
|---|---|
| **Name** | Pre-Convolutional ViT (PCViT) |
| **Reference** | Adapted from Yang et al., *Energies* 2022 |
| **Task** | 24-class NIR spectroscopy classification (12 coal, 12 gangue) |
| **Input** | (N, 3, 1500) tensor: SNV channel, 1st derivative, 2nd derivative of 1500-band spectra |
| **Output** | (N, 24) logits |
| **Parameters** | 903,576 |
| **Weights file** | `pcvit_expert.pth` |
| **SHA-256** | `ca950a2e78b885849916efc9ce1b86e63c50356544bd9bd9049eca102ccc8610` |
| **Inference** | CPU-only, `map_location="cpu"`, `weights_only=True` |
| **Training seed** | 42 |

## Intended Use

Assist human operators in sorting coal from gangue rock on a mining conveyor belt by classifying
near-infrared reflectance spectra. **Not a standalone safety system.** Any REJECT or REVIEW decision
should be confirmed by a human operator before actuation.

## Training Data

- **Source**: Laboratory NIR spectra collected at Chinese University of Mining and Technology and
  affiliated Chinese coal mines (Shandong, Anhui, Shanxi provinces).
- **Size**: 10,497 unique spectra (10,520 raw; 23 exact duplicates removed).
- **Split**: Stratified random, seed 42. Train: 7,353 / Val: 1,578 / Test: 1,566.
- **Classes**: 12 coal types, 12 gangue types (see `backend/assets/class_list.json`).

## Preprocessing Pipeline (fixed, immutable)

1. Validate exactly 1505 raw numeric values or 1500 clean values. The raw layout retains five numeric metadata values and the first 1495 reflectance bands in the model input.
2. Refuse non-finite, out-of-range or flat reflectance scans. Front-crop to 1500 values as in notebooks/coalrock_preprocess2.ipynb.
3. Standard Normal Variate (SNV) normalization per scan.
4. Savitzky-Golay derivatives: window=11, polyorder=2, orders 1 and 2 on the SNV channel.
5. Stack: (N, 3, 1500).

## Test Set Performance (1,566 spectra, all numbers from `artifacts/metrics.json`)

### Classification Metrics (95% CI via stratified bootstrap, 2000 resamples)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | **95.72%** | [94.70%, 96.62%] |
| Macro F1 | **95.50%** | [94.37%, 96.43%] |
| Weighted F1 | 95.72% | — |
| Balanced Accuracy | 95.59% | — |
| Cohen's κ | 0.9553 | — |
| MCC | 0.9554 | — |
| Top-2 Accuracy | 98.91% | — |
| Top-3 Accuracy | 99.36% | — |

### Error Decomposition

| Error Type | Count |
|---|---|
| Coal predicted as coal (correct) | 741 |
| Rock predicted as rock (correct) | 820 |
| **Coal predicted as rock (false reject)** | **4** |
| **Rock predicted as coal (safety miss)** | **1** |

### Safety Metrics — Binary View (gangue = positive class, 95% Clopper–Pearson exact CI)

| Metric | Value | 95% CI |
|---|---|---|
| Gangue Recall (Sensitivity) | **99.88%** | [99.32%, 100.0%] |
| Miss Rate | **0.12%** | [0.003%, 0.677%] |
| Specificity | 99.46% | [98.63%, 99.85%] |
| Precision | 99.51% | — |
| F2 Score | 99.81% | — |
| ROC-AUC | 99.996% | — |
| PR-AUC | 99.997% | — |

> One rock scan among 821 rock scans was called coal. The two-sided exact 95% interval for the miss rate is approximately 0.003%–0.677%, calculated by complementing gangue_recall_ci_95 in artifacts/metrics.json.

### Per-class weaknesses

Source — artifacts/metrics.json, per_class.

| Class | Recall | Precision | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Anthracite coal grade 1 | 0.871 | 0.982 | 0.923 | 62 |
| Arenaceous shale 1 | 0.885 | 0.902 | 0.893 | 52 |
| Black shale 1 | 0.923 | 0.889 | 0.906 | 52 |
| Fat coal | 0.887 | 0.982 | 0.932 | 62 |
| Lean coal | 0.919 | 0.966 | 0.942 | 62 |
| Lean-thin coal | 0.903 | 0.966 | 0.933 | 62 |
| Lignite coal grade 1 | 0.903 | 0.918 | 0.911 | 62 |

## Safety Override (Conveyor Logic)

Batch decision uses mean-pooled probabilities across all scans. If **≥ 4 scans** have total gangue
probability mass ≥ **0.95**, the verdict is forced to REJECT regardless of the batch mean.

- Threshold (0.95) and k (4) chosen by expected-cost sweep on validation data only (cost ratio
  assumed: miss costs 10× false alarm).
- Unvalidated input changes an otherwise ACCEPT verdict to REVIEW. REJECT has precedence. Source — backend/assets/safety_config.json and backend/main.py.

## Baselines (single seed, SNV channel only; from `artifacts/baselines.json`)

| Model | Test Accuracy |
|---|---|
| PCViT (this model) | **95.72%** |
| Random Forest (50 trees) | 85.89% |
| 1-Nearest-Neighbour | 77.27% |
| RBF-SVM | 77.27% |
| PLS-DA (15 components) | 67.24% |

PCViT outperforms the strongest baseline (Random Forest) by +9.8 percentage points.
Run `ml/baselines.py` to reproduce (single seed, baselines on SNV channel only).

**Leakage probe**: 7.6% of test spectra have a training-set neighbour with cosine similarity > 0.999
in SNV space. The 1-NN baseline at 77.27% indicates the model is learning something beyond
memorisation, but the split does not fully control within-specimen correlation.


## Efficiency

- **Parameters**: 903,576
- **Forward pass latency (Colab CPU)**: P50 ~16 ms, P99 ~25.7 ms (single batch, model only;
  not including CSV parse or preprocessing)
- **Scope**: These numbers are for the model forward pass only, not end-to-end API latency.
  CSV parsing typically dominates for large files.

## Known Limitations

See `docs/LIMITATIONS.md` for a full account. Key points:
- Row-level split; within-specimen correlation not controlled.
- Laboratory data only; not validated on a moving belt or under field conditions.
- Thresholds are assumptions; recalibrate for each deployment site.
- Confidence values are uncalibrated softmax outputs.

## Evidence scope

95.72% accuracy on 1,566 held-out test scans, with coal-to-coal / rock-to-rock / coal-to-rock / rock-to-coal counts of 741 / 820 / 4 / 1. One rock scan was called coal. Source — artifacts/metrics.json, multi_class.accuracy and error_decomposition.

The random per-scan split can place scans from the same physical specimens in training and testing. Underground field validation remains outstanding. Architecture and training details — notebooks/coalrock_training_evaluation.ipynb. Split counts — notebooks/coalrock_preprocess2.ipynb. Baseline comparisons — artifacts/baselines.json.
