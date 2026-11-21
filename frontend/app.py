"""
frontend/app.py — Spectra-Sense Dashboard v4
Streamlit UI for the PCViT coal-gangue classifier. Upload flow + backend calls live here;
all drawing lives in render.py.
"""
import hashlib
import os

import requests
import streamlit as st

from render import DEFAULT_FUEL_CLASSES, render_result

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spectra-Sense | Coal-Gangue Classifier", page_icon="⛏️", layout="wide")

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (KeyError, FileNotFoundError):
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_URL = BACKEND_URL.rstrip("/")

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "class_images")
PREDICT_TIMEOUT_S = 300


class BackendError(Exception):
    """A user-presentable backend failure."""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fuel_classes(url: str):
    """Fuel/gangue boundary comes from the backend (single source of truth); local list is a fallback."""
    try:
        r = requests.get(f"{url}/meta", timeout=10)
        r.raise_for_status()
        return r.json()["fuel_classes"]
    except Exception:
        return DEFAULT_FUEL_CLASSES


@st.cache_data(show_spinner=False, max_entries=8)
def call_backend(url: str, name: str, content: bytes, _digest: str):
    """POST the file once per distinct upload (Streamlit reruns must not re-run inference)."""
    try:
        r = requests.post(f"{url}/predict", files={"file": (name, content, "text/csv")}, timeout=PREDICT_TIMEOUT_S)
    except requests.exceptions.ConnectionError:
        raise BackendError("Cannot reach the backend. If running locally, start `uvicorn main:app` on port 8000; "
                           "if deployed, check the Render service logs.")
    except requests.exceptions.Timeout:
        raise BackendError(f"Request timed out (>{PREDICT_TIMEOUT_S}s). The backend may be waking up — try again.")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text[:200])
        except ValueError:
            detail = r.text[:200]
        raise BackendError(f"Backend error {r.status_code}: {detail}")
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("⛏️ Spectra-Sense")
st.caption("Low-latency Coal-Gangue Classification via NIR Spectroscopy | PCViT | IIT (BHU)")
st.markdown("---")
st.info(
    "ℹ️ **Cold start notice:** the backend runs on Render's free tier. After 15 minutes idle the first request "
    "can take 30–60 seconds while the instance wakes up; later requests are much faster.",
    icon="🕐",
)

with st.expander("How this is used in a mine"):
    st.write("Sample check — a lump is scanned and gets a verdict.")
    st.write("Belt-segment screening — a file of many scans is screened, and a cluster of gangue scans gives REJECT.")
    st.write("Lithology and grade passport — identification among 24 classes with origin and geochemistry.")
    st.write("Decision support, validated on a lab spectral library, not yet validated under underground conditions.")
    st.caption("Class count source — backend/assets/class_list.json")

uploaded_file = st.file_uploader(
    "Upload a spectrometer CSV file",
    type=["csv"],
    help="Dataset-format CSVs are the validated input. Clean 1500-band spectra are accepted but flagged as unvalidated. Max 10 MB.",
)

if uploaded_file is not None:
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()

    # Wake-up ping: a sleeping Render instance only starts booting when it receives a request.
    try:
        requests.get(f"{BACKEND_URL}/health", timeout=5)
    except requests.exceptions.RequestException:
        pass

    with st.spinner(
        "Running inference… The first request after idle may take up to a minute while the backend wakes up. "
        "Large files can take about a minute on the free tier."
    ):
        try:
            data = call_backend(BACKEND_URL, uploaded_file.name, content, digest)
        except BackendError as e:
            st.error(str(e))
            st.stop()

    render_result(data, fetch_fuel_classes(BACKEND_URL), IMAGES_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER — every figure is traceable to docs/CLAIMS_LEDGER.md
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Pre-Convolutional ViT (PCViT, 903,576 parameters) · held-out lab test accuracy 95.72% (95% CI 94.70–96.62%) across 24 lithologies · "
    "1 rock→coal boundary error in 1,566 test scans · model forward-pass P99 25.7 ms on Colab CPU (not end-to-end) · "
    "research prototype on lab-measured spectra, not validated on a live belt · "
    "Amit Kumar, B.Tech Mining, IIT (BHU)</small>",
    unsafe_allow_html=True,
)

st.caption("Performance source — artifacts/metrics.json, counts 741 / 820 / 4 / 1; one rock scan called coal. Model and latency source — notebooks/coalrock_training_evaluation.ipynb.")
