# Deployment instructions — release candidate

These instructions describe the commands and configuration in this repository. The Deploy steps section lists the selected hosting settings.

## Python and dependencies

VERIFIED — Both [.python-version](../.python-version) and [backend/.python-version](../backend/.python-version) specify Python 3.12. The Render setting below selects 3.12.9.

From the repository root, in the intended Python environment:

```sh
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

These are the README install commands expressed from the root. [backend/requirements.txt](../backend/requirements.txt), lines 8–12, installs `torch==2.11.0+cpu` with `--extra-index-url https://download.pytorch.org/whl/cpu`. [frontend/requirements.txt](../frontend/requirements.txt) pins the tested frontend versions: `streamlit==1.67.0`, `requests==2.34.2`, `matplotlib==3.11.2`, `numpy==2.0.2`, `Pillow==13.0.0`. Transitive dependencies are resolved separately.

## Backend command and checkpoint

VERIFIED — Working directory must be `backend/` for the downloader's relative `models/` path. The exact deployment start command appears in the [README deployment section](../README.md#deployment-render), [backend/download_model.py](../backend/download_model.py) and [backend/main.py](../backend/main.py):

```sh
python download_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT
```

This is a shell command using POSIX `$PORT` expansion, not PowerShell syntax. The process environment must supply the port. The README local start command, also from `backend/`, is:

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

VERIFIED — The downloader requests `pcvit_expert.pth` from Hugging Face repository `amit2-0kr/spectra-sense` and saves it under `backend/models/` when invoked as above. It skips downloading an existing file and supplies no revision argument. The checkpoint is not shipped in this candidate. The backend loads it from a path anchored to `backend/main.py`, refuses missing weights, and verifies SHA-256 before loading:

```text
ca950a2e78b885849916efc9ce1b86e63c50356544bd9bd9049eca102ccc8610
```

Source: backend/download_model.py:25–50; backend/main.py:42–52,90–104. `EXPECTED_WEIGHTS_SHA256` is a code constant, not an environment setting. NOT RUN — remote checkpoint availability and hash.

## Frontend entry and API URL

VERIFIED — The entry file is [frontend/app.py](../frontend/app.py). The README command is run from `frontend/`:

```sh
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

This local example uses shell `export` syntax. For a hosted frontend, use the actual backend URL once it is known. app.py:19–23 first reads `st.secrets["BACKEND_URL"]`; on KeyError or FileNotFoundError it reads environment variable `BACKEND_URL`, defaulting to `http://localhost:8000`. It removes trailing slashes. A configured secret takes precedence over the environment variable. The README names the same key for Streamlit Cloud secrets.

The frontend resolves its images relative to its own file under `frontend/assets/class_images/` (app.py:25). The prediction request timeout is 300 seconds (app.py:26).

## Configuration read by the application

| Setting | Source and behavior |
| --- | --- |
| PORT | Deployment shell start command; no value specified in repo. |
| BACKEND_URL | Frontend secret first, environment fallback, then http://localhost:8000. |
| LOG_LEVEL | backend/main.py:28; default INFO. |
| MAX_UPLOAD_MB | backend/main.py:31; integer default 10, multiplied by 1024 * 1024. |
| TORCH_NUM_THREADS | backend/main.py:54; integer default 1. |
| RENDER_GIT_COMMIT | backend/main.py:137; preferred revision label for /version. |
| GIT_SHA | Same line; fallback revision label, otherwise "unknown". |
| PYTHON_VERSION | Set to 3.12.9 for Render, as listed below. Application code does not read it. |
| SPECTRASENSE_RAW_DIR | tests/test_raw_corpus.py:15–17; optional external corpus path, otherwise its 24 cases skip. |
| SPECTRASENSE_INPUTS | ml/run_evaluation.ipynb; optional evaluation-array directory, default repo inputs/. Not needed for serving. |

VERIFIED — These settings are evidenced in the named repository files. No additional hosting environment variables are asserted here.

## Verification and missing-weight limitation

VERIFIED — backend/main.py defines /health, /ready, /version, /meta and /predict. /version includes the loaded weights SHA-256. These route definitions are not evidence of a live deployment.

See the [README test instructions](../README.md#3-tests) for the weight-free run and its skip reasons. Tests that start the real backend require the checkpoint, and the demo and corpus regressions use separately supplied data.

## Demo files

The demo pack is kept outside this repository. Upload the files from your laptop through the Streamlit file uploader. The verdicts below were recorded for these filenames during the local demo checks.

| File | Recorded verdict |
| --- | --- |
| `S01_0.3_coking_coal_1scan.csv` | ACCEPT |
| `S01_Arenaceous_shale_1_1scan.csv` | REJECT |
| `S02_0.3_coking_coal_3scans.csv` | ACCEPT |
| `S03_Anthracite_coal_grade_1_Carbonaceous_shale_5scans_1rock.csv` | REVIEW |
| `S03_Anthracite_coal_grade_1_Carbonaceous_shale_10scans_4rock.csv` | REJECT (Safety Override) |
| `S04_belt_20scans_2rock.csv` | REVIEW |
| `S04_belt_21scans_1rock.csv` | ACCEPT (known limit demonstration) |
| `S05_01_model_calls_rock_coal_Black_shale_1_scan470.csv` | REVIEW |

## Deploy steps

### Render web service

| Setting | Value |
| --- | --- |
| Plan | Free |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python download_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Environment variable | `PYTHON_VERSION=3.12.9` |
| Health check path | `/health` |

### Streamlit Cloud

| Setting | Value |
| --- | --- |
| Branch | `main` |
| Main file | `frontend/app.py` |
| Python | 3.12 |
| Secret | Set `BACKEND_URL` to the Render service URL. |

Other hosting settings and the actual Render service URL are unknown.
