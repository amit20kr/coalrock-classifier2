# Spectra-Sense — PCViT Coal-Gangue Classifier

Low-latency classification of coal vs. waste rock (gangue) from NIR spectra (1000–2499 nm) with a Pre-Convolutional ViT (PCViT, 903,576 parameters).

**Team:** Amit Kumar | Roll 23155012 | B.Tech Mining Engineering, IIT (BHU)  
**Supervisor:** Dr. Aishwarya Mishra  
**Hackathon:** Bharat Academix CodeQuest (Unstop) — Round 2

> Research prototype trained and evaluated on lab-measured spectra. Not validated on a live conveyor belt.
> See [Limitations](docs/LIMITATIONS.md) and the evidence sources below.

[Model card](docs/MODEL_CARD.md)

---

## The Physics (hypothesis)

Coal is largely amorphous carbon with a comparatively featureless NIR spectrum. Gangue such as clay and shale can contain Al-OH bearing minerals (kaolinite, illite) that absorb near ~2200 nm, so the 2nd-derivative channel was built to make that region easy to see.
Which spectral cues the trained model actually relies on has **not yet been verified** (explainability analysis is pending); the dashboard therefore shows the 1400/1900/2200 nm regions as reference windows, not as detections.

---

## Quick Start (Local)

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python download_model.py      # pulls pcvit_expert.pth from HuggingFace (or place it in backend/models/)
uvicorn main:app --host 0.0.0.0 --port 8000
```

Check it: http://localhost:8000/health · http://localhost:8000/ready · http://localhost:8000/meta · http://localhost:8000/version

### 2. Frontend

```bash
cd frontend
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

Open http://localhost:8501 and upload a spectrometer `.csv`.

### 3. Tests

From the repository root, use a Python 3.12 environment:

```bash
python -m pip install -r backend/requirements.txt
python -m pip install pytest httpx
python -m pytest -q -rs
```

The same command works without the checkpoint, test arrays, or demo CSVs. Pytest reports the missing prerequisites:

- **Checkpoint** (`backend/models/pcvit_expert.pth`). The API, API contract, safety, safe configuration, checkpoint integrity, and model checkpoint tests skip, as does the API verdict comparison for altered trailing bands. The other parser and preprocessing tests still run.
- **Demo CSV** (`Demo test/Carbonaceous shale.csv`). The real-file regression needs both this CSV and the checkpoint.
- **Raw corpus** (`SPECTRASENSE_RAW_DIR`). Leave unset in a data-free checkout. To run the 24 corpus cases, set it to the directory containing the 24 original class CSVs.
- **Frontend dependencies**. The frontend test module skips if Streamlit's testing module is unavailable. Install `frontend/requirements.txt` to include these tests.

`SPECTRASENSE_INPUTS` selects the directory containing `X_test.npy`, `y_test.npy`, and `label_encoder.pkl` for `ml/run_evaluation.ipynb`. It defaults to `inputs/` and is used for notebook evaluation. Pytest does not require these arrays.

To run the checkpoint tests, obtain `pcvit_expert.pth` from Hugging Face (`amit2-0kr/spectra-sense`) using the backend setup above.

---

## Input format (important)

| Layout | Status |
|---|---|
| **Raw dataset layout** (rows = bands, one column per scan, 5 numeric metadata rows + text rows, wavelength in column 0) | **Validated** — the layout the model was trained and tested on |
| Clean 1500-band spectra (no metadata rows) | Accepted but flagged `UNVALIDATED` in the UI; the model is markedly less reliable on it (see `docs/LIMITATIONS.md` §8) |

Limits: 10 MB per upload, 2000 scans per file. Nothing is subsampled — every scan reaches the model.

---

## Safety rule (versioned in `backend/assets/safety_config.json`)

- A scan is **flagged** when its *total* gangue probability (sum over the 12 rock classes) is ≥ 0.95.
- **REJECT** if the batch-mean gangue probability is ≥ 0.5.
- **REJECT (Safety Override)** if the batch mean says coal but **≥ 4** scans are flagged.
- **REVIEW** if the batch is called coal but mean top-class probability is below 0.60 (heuristic, uncalibrated).
- In a window at or below `review_any_flagged_up_to_scans` (default 20), a single flagged scan changes an otherwise ACCEPT verdict to REVIEW. The warning gives the flagged scan count and 0.95 threshold. REJECT retains precedence.
- Above the cutoff, 1–3 flagged scans raise a warning on an otherwise ACCEPT batch. See the [large-file injection test and cutoff limits](docs/LIMITATIONS.md#9-what-the-k--4-rule-does-and-does-not-catch-measured-real-data).

---

## Demo

Rename a rock file (e.g. `Carbonaceous shale.csv`) to `PURE_ANTHRACITE_A_GRADE.csv` and upload it. The verdict is REJECT: the model sees spectra, not file names.

---

## Architecture

```
(Batch, 3, 1500)       3-channel NIR tensor [SNV | 1st Deriv | 2nd Deriv]
       ↓
[CNN Stem]             3 conv layers, stride-2 → compresses 1500 → 375 tokens
       ↓
[CLS Token + Pos Emb]  Prepend learnable summary vector
       ↓
[Transformer × 4]      4 heads, 4 layers — global context across all 375 tokens
       ↓
[CLS → Linear]         24-class logit vector
```

Model size: 903,576 parameters | 3.6 MB.

---

## Key Results (held-out lab test set, 1,566 scans)

| Metric | Value |
|---|---|
| 24-class accuracy | 95.72% (95% CI 94.70–96.62%) |
| Gangue recall | 99.88% (95% CI 99.32–100%) |
| Rock→coal boundary errors | 1 of 1,566 |
| Coal→rock false rejects | 4 |
| Model forward-pass latency | P99 25.7 ms (Colab CPU; model only, not end-to-end API) |

Scans in the test set come from the same measurement files as training (deduplicated); generalisation to new sites or sensors is untested.

---

## Scope and limits

Decision-support demo using a lab spectral library. 24 classes, 1000–2500 nm, 1500 bands. Held-out accuracy 95.72% on 1,566 test scans with one rock scan called coal. Unvalidated input goes to REVIEW if the result would otherwise be ACCEPT.

The fixed-distance lab replay examined single scans and simulated small windows on held-out scans from the same specimens as training. For sensor use, the REVIEW cutoff must be at least the maximum scans per lump the sensor delivers. Real lumps, distance variation, moisture, dust and new seams remain untested. Measure the REVIEW rate in a pilot. [Few-scan evaluation](docs/FEW_SCAN_EVALUATION.md).

See [held-out metrics](artifacts/metrics.json), [class names](backend/assets/class_list.json), and [limitations](docs/LIMITATIONS.md).

## Data and credits

- Dataset: Lv, Wang, Yang and Ge, *Scientific Data* (2024). [Paper](https://doi.org/10.1038/s41597-024-03422-w) and [Zenodo dataset record](https://doi.org/10.5281/zenodo.11137126).
- The Pre-Convolutional Vision Transformer (PCViT) is adapted from Yang et al., *Energies* (2022). See the [model card](docs/MODEL_CARD.md).

## Deployment (Render)

- Start command (from `backend/`): `python download_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Python version:** pinned to 3.12 via `.python-version` (also set `PYTHON_VERSION=3.12.x` in the Render dashboard). Render's default for new 2026 services is ≥ 3.13.
- **torch** is pinned to `2.11.0+cpu`, the version printed by the training notebook.
- Checkpoint integrity is enforced at startup using the expected SHA-256 in backend/main.py.
- **Memory:** the free tier has 512 MB. A 5 MB raw file peaks near 430–500 MB in tests; use a 1 GB instance for demos with large files.
- Frontend: set `BACKEND_URL` in Streamlit Cloud secrets. The free-tier backend cold-starts in 30–60 s; the UI shows a notice.
- Live URLs: _add the Render and Streamlit URLs here after checking the deployed services._

## Evidence and current input policy

95.72% accuracy on 1,566 held-out test scans, with coal-to-coal / rock-to-rock / coal-to-rock / rock-to-coal counts of 741 / 820 / 4 / 1. One rock scan was called coal. Source — artifacts/metrics.json, multi_class.accuracy and error_decomposition.

Unvalidated clean input changes an otherwise ACCEPT result to REVIEW. REJECT retains precedence. Split and preprocessing source — notebooks/coalrock_preprocess2.ipynb. Model architecture and CPU latency source — notebooks/coalrock_training_evaluation.ipynb. Safety thresholds — backend/assets/safety_config.json and artifacts/batch_metrics.json.
