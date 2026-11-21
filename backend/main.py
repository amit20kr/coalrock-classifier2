"""
main.py — FastAPI inference engine for the Spectra-Sense PCViT coal/gangue classifier.

Endpoints: /predict, /health (liveness), /ready (model warm), /version, /meta.
Render start command (run from backend/):  python download_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

# Works both as `uvicorn main:app` (run inside backend/) and as `backend.main` (pytest from repo root).
try:
    from .model import PCViT
    from .pipeline import FUEL_CLASSES, XRF_PASSPORT, SpectraFormatError, load_class_list, load_spectra
except ImportError:  # no parent package
    from model import PCViT
    from pipeline import FUEL_CLASSES, XRF_PASSPORT, SpectraFormatError, load_class_list, load_spectra

logger = logging.getLogger("spectra_sense")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

APP_VERSION = "2.0.0"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

app = FastAPI(
    title="Spectra-Sense Inference Engine",
    description="PCViT coal-gangue classification from NIR spectra (1000-2499 nm).",
    version=APP_VERSION,
)

# ──────────────────────────────────────────────────────────────────────────────
# BOOT: everything is loaded once. Paths are anchored to this file, not the cwd.
# ──────────────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent
_CLASS_LIST_PATH = _BASE / "assets" / "class_list.json"
_SAFETY_CFG_PATH = _BASE / "assets" / "safety_config.json"
_MODEL_PATH = _BASE / "models" / "pcvit_expert.pth"

for _p, _hint in (
    (_CLASS_LIST_PATH, "It is required for class-name mapping."),
    (_MODEL_PATH, "Run `python download_model.py` first (see the Render start command)."),
):
    if not _p.exists():
        raise FileNotFoundError(f"{_p.name} not found at {_p}. {_hint}")

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
device = torch.device("cpu")

CLASSES = load_class_list(str(_CLASS_LIST_PATH))
NUM_CLASSES = len(CLASSES)

def _review_unvalidated(config):
    value = config.get("review_unvalidated_input", True)
    if value is False:
        raise RuntimeError("review_unvalidated_input must be enabled at startup")
    return True


# Safety-net parameters come from a versioned config, never hard-coded in logic.
_cfg = {"version": "default", "gangue_mass_threshold": 0.95, "min_flagged_scans": 4, "chunk_size": 8,
        "review_confidence": 0.60, "review_unvalidated_input": True,
        "review_any_flagged_up_to_scans": 20}
if _SAFETY_CFG_PATH.exists():
    _cfg.update(json.loads(_SAFETY_CFG_PATH.read_text(encoding="utf-8")))
GANGUE_OVERRIDE_THRESHOLD = float(_cfg["gangue_mass_threshold"])   # per-scan TOTAL gangue probability
GANGUE_OVERRIDE_MIN_SCANS = int(_cfg["min_flagged_scans"])         # k scans above threshold => reject
CHUNK_SIZE = int(_cfg["chunk_size"])                               # bounds peak memory per forward()
REVIEW_CONFIDENCE = float(_cfg["review_confidence"])               # ACCEPT -> REVIEW below this (heuristic, uncalibrated)
REVIEW_UNVALIDATED_INPUT = _review_unvalidated(_cfg)  # if true: never ACCEPT an unvalidated input layout
REVIEW_ANY_FLAGGED_UP_TO_SCANS = int(_cfg["review_any_flagged_up_to_scans"])
SAFETY_CONFIG_VERSION = str(_cfg["version"])

gangue_indices = [i for i, c in enumerate(CLASSES) if c not in FUEL_CLASSES]
fuel_indices = [i for i, c in enumerate(CLASSES) if c in FUEL_CLASSES]

# Fail loudly at startup if the fuel/gangue boundary is not what the pitch says it is.
_missing = [c for c in FUEL_CLASSES if c not in CLASSES]
if _missing or len(fuel_indices) != 12 or len(gangue_indices) != 12:
    raise RuntimeError(
        f"FUEL_CLASSES/class_list mismatch: unmatched={_missing}, fuel={len(fuel_indices)}, "
        f"gangue={len(gangue_indices)} (expected 12/12)."
    )

EXPECTED_WEIGHTS_SHA256 = "ca950a2e78b885849916efc9ce1b86e63c50356544bd9bd9049eca102ccc8610"


def _verify_weights(path):
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError("pcvit_expert.pth checksum mismatch. Refusing to start.")
    return checksum


WEIGHTS_SHA256 = _verify_weights(_MODEL_PATH)

model = PCViT(num_classes=NUM_CLASSES)
try:
    model.load_state_dict(torch.load(str(_MODEL_PATH), map_location=device, weights_only=True))
except RuntimeError as e:
    raise RuntimeError(
        "model.py's PCViT architecture does not match the saved checkpoint. Attribute names must stay "
        f"exactly stem/cls/pos/tf/head and num_classes must be {NUM_CLASSES}. Original error: {e}"
    ) from e
model.eval()

# One warm-up pass so the first real request does not pay initialisation cost.
with torch.inference_mode():
    model(torch.zeros(1, 3, 1500))
_ready = True
_infer_lock = threading.Lock()   # serialise heavy work on the 512 MB free tier; /health stays responsive


# ──────────────────────────────────────────────────────────────────────────────
# OPS ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness. Cheap and independent of inference."""
    return {"status": "ok", "model": "PCViT", "num_classes": NUM_CLASSES, "device": str(device)}


@app.get("/ready")
async def ready():
    return {"ready": _ready}


@app.get("/version")
async def version():
    return {
        "app_version": APP_VERSION,
        "git_sha": os.getenv("RENDER_GIT_COMMIT", os.getenv("GIT_SHA", "unknown")),
        "weights_sha256": WEIGHTS_SHA256,
        "safety_config_version": SAFETY_CONFIG_VERSION,
        "torch": torch.__version__,
    }


@app.get("/meta")
async def meta():
    """Single source of truth for class lists and the safety rule (the frontend reads this)."""
    return {
        "classes": CLASSES,
        "fuel_classes": [CLASSES[i] for i in fuel_indices],
        "gangue_classes": [CLASSES[i] for i in gangue_indices],
        "safety": {
            "gangue_mass_threshold": GANGUE_OVERRIDE_THRESHOLD,
            "min_flagged_scans": GANGUE_OVERRIDE_MIN_SCANS,
            "review_confidence": REVIEW_CONFIDENCE,
            "review_any_flagged_up_to_scans": REVIEW_ANY_FLAGGED_UP_TO_SCANS,
            "rule": (
                "REJECT if the batch-mean gangue probability >= 0.5, or if at least "
                f"{GANGUE_OVERRIDE_MIN_SCANS} scans each have total gangue probability >= "
                f"{GANGUE_OVERRIDE_THRESHOLD}. Otherwise REVIEW for low confidence, an unvalidated layout, "
                f"or at least one flagged scan in a window of {REVIEW_ANY_FLAGGED_UP_TO_SCANS} scans or fewer. "
                "Above that cutoff, flagged scans below the override count raise a warning."
            ),
        },
        "limits": {"max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024), "max_scans": 2000},
    }


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT (sync on purpose: FastAPI runs it in a worker thread so /health is never blocked)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    rid = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()
    try:
        contents = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Payload Too Large: the maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        t_read = time.perf_counter()

        ingest = load_spectra(contents)
        tensor = ingest.tensor                       # (N, 3, 1500)
        n_scans = tensor.shape[0]
        t_prep = time.perf_counter()

        sum_probs = torch.zeros(NUM_CLASSES)
        gangue_mass_chunks = []
        with _infer_lock, torch.inference_mode():
            for start in range(0, n_scans, CHUNK_SIZE):
                probs = torch.softmax(model(tensor[start:start + CHUNK_SIZE]), dim=1)   # (chunk, 24)
                sum_probs += probs.sum(dim=0)
                gangue_mass_chunks.append(probs[:, gangue_indices].sum(dim=1))          # TOTAL gangue mass
        t_infer = time.perf_counter()

        per_scan_mass = torch.cat(gangue_mass_chunks)                 # (N,)
        mean_probs = sum_probs / n_scans
        mean_gangue_mass = float(mean_probs[gangue_indices].sum())
        n_flagged = int((per_scan_mass >= GANGUE_OVERRIDE_THRESHOLD).sum())

        # Family decision first (ties go to the safe side), then the minimum-evidence override.
        family_is_gangue = mean_gangue_mass >= 0.5 - 1e-6
        override = (not family_is_gangue) and n_flagged >= GANGUE_OVERRIDE_MIN_SCANS
        is_gangue = family_is_gangue or override

        family_idx = gangue_indices if family_is_gangue else fuel_indices
        top_in_family = family_idx[int(mean_probs[family_idx].argmax())]
        top5_probs, top5_idx = mean_probs.topk(5)

        confidence = float(mean_probs[top_in_family])
        warnings = list(ingest.notes)
        if override:
            predicted_class, verdict = "Gangue Contamination Detected", "REJECT (Safety Override)"
        elif family_is_gangue:
            predicted_class, verdict = CLASSES[top_in_family], "REJECT"
        else:
            predicted_class, verdict = CLASSES[top_in_family], "ACCEPT"
            # REVIEW can only soften an ACCEPT, never a REJECT.
            if confidence < REVIEW_CONFIDENCE:
                verdict = "REVIEW"
                warnings.append(
                    f"Mean top-class confidence {confidence:.2f} is below {REVIEW_CONFIDENCE:.2f}; "
                    "a human check is recommended before accepting this batch."
                )
            elif REVIEW_UNVALIDATED_INPUT and not ingest.validated:
                verdict = "REVIEW"
                warnings.append("Input layout is outside the validated range, so the batch cannot be auto-accepted.")

        if verdict == "ACCEPT" and n_flagged > 0 and n_scans <= REVIEW_ANY_FLAGGED_UP_TO_SCANS:
            verdict = "REVIEW"
            warnings.append(
                f"{n_flagged} scan(s) met the gangue probability threshold ({GANGUE_OVERRIDE_THRESHOLD:.2f}); "
                f"manual review is required for windows of {REVIEW_ANY_FLAGGED_UP_TO_SCANS} scans or fewer."
            )
        elif not is_gangue and 0 < n_flagged < GANGUE_OVERRIDE_MIN_SCANS:
            warnings.append(
                f"{n_flagged} scan(s) exceeded the gangue threshold ({GANGUE_OVERRIDE_THRESHOLD:.2f}) but at least "
                f"{GANGUE_OVERRIDE_MIN_SCANS} are required to reject the batch. Inspect these scans manually."
            )

        # Plot the most gangue-like scan when any scan is flagged, otherwise scan 0.
        plot_idx = int(per_scan_mass.argmax()) if n_flagged > 0 else 0
        passport = XRF_PASSPORT.get(predicted_class, {"Origin": "Standard Mining Library", "Geochemistry": {}})
        t_end = time.perf_counter()

        logger.info(
            "rid=%s scans=%d fmt=%s verdict=%s flagged=%d total_ms=%.0f",
            rid, n_scans, ingest.input_format, verdict, n_flagged, (t_end - t_start) * 1000,
        )
        return {
            # ---- v1 keys (unchanged) ----
            "verdict": verdict,
            "is_gangue": is_gangue,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_5": [
                {"class": CLASSES[i.item()], "probability": float(p.item())}
                for p, i in zip(top5_probs, top5_idx)
            ],
            "digital_passport": passport,
            "spectral_channels": {
                "wavelengths": ingest.wavelengths,
                "snv": tensor[plot_idx, 0].tolist(),
                "second_derivative": tensor[plot_idx, 2].tolist(),
            },
            # ---- v2 additions ----
            "request_id": rid,
            "plot_scan_index": plot_idx,
            "scan_summary": {
                "n_scans": n_scans,
                "n_flagged": n_flagged,
                "flag_threshold": GANGUE_OVERRIDE_THRESHOLD,
                "min_flagged_to_reject": GANGUE_OVERRIDE_MIN_SCANS,
                "mean_gangue_mass": round(mean_gangue_mass, 4),
                "max_gangue_mass": round(float(per_scan_mass.max()), 4),
                "per_scan_gangue_mass": [round(float(x), 4) for x in per_scan_mass.tolist()],
            },
            "input": {"format": ingest.input_format, "validated": ingest.validated, "dropped": ingest.dropped},
            "warnings": warnings,
            "timing_ms": {
                "read": round((t_read - t_start) * 1000, 1),
                "preprocess": round((t_prep - t_read) * 1000, 1),
                "inference": round((t_infer - t_prep) * 1000, 1),
                "total": round((t_end - t_start) * 1000, 1),
            },
            "model": {"version": APP_VERSION, "weights_sha256": WEIGHTS_SHA256[:12],
                      "safety_config": SAFETY_CONFIG_VERSION},
        }

    except HTTPException:
        raise
    except SpectraFormatError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("rid=%s unexpected failure", rid)   # detail stays in the server log only
        raise HTTPException(status_code=500, detail=f"Inference failed (request id {rid}). Please retry or contact the operator.")
