"""
pipeline.py — The Data Layer
Orientation + wavelength-axis handling + 3-Channel Physics Tensor Constructor.

TRAIN/SERVE PARITY (verified 2026-11 against the real dataset files):
the model was trained on spectra produced by the notebook's `surgical_load`, which on the
raw dataset layout (1507 rows x N cols: 5 numeric metadata rows, 2 text rows, 1500 band rows,
wavelength in column 0) keeps the 5 numeric metadata values as the FIRST 5 "bands" and
front-crops to 1500. This module reproduces that exactly for the dataset layout and flags every
other layout as `validated=False` because the checkpoint is measurably less reliable on it.
"""
import io
import json
import pickle
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

EXPECTED_BANDS = 1500

# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: 12 FUEL classes vs 12 GANGUE classes
# This list is the boundary the safety net enforces.
# ──────────────────────────────────────────────────────────────────────────────
FUEL_CLASSES = [
    "Anthracite coal grade 1",
    "Anthracite coal grade 2",
    "Lean coal",
    "Lean-thin coal",
    "Thin coal",
    "Coking coal",
    "0.3 coking coal",
    "Fat coal",
    "Gas-fat coal",
    "Gas coal",
    "Lignite coal grade 1",
    "Lignite coal grade 2",
]

# ──────────────────────────────────────────────────────────────────────────────
# XRF PASSPORT — nested structure (Origin + Geochemistry)
# Geochemistry values sourced from backend/assets/XRF_database.csv (real XRF measurements).
# Frontend calls: passport_data.get('Origin', ...) and passport_data.get('Geochemistry', {})
# ──────────────────────────────────────────────────────────────────────────────
XRF_PASSPORT = {
    # ── FUEL CLASSES (coal — only SiO2, Al2O3, Fe2O3 measurable via XRF) ─────
    "Anthracite coal grade 1": {
        "Origin": "Jincheng Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "2.63%", "Al2O3": "1.01%", "Fe2O3": "0.87%"
        }
    },
    "Anthracite coal grade 2": {
        "Origin": "Yangquan Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "4.61%", "Al2O3": "1.17%", "Fe2O3": "0.81%"
        }
    },
    "Lean coal": {
        "Origin": "Fengfeng Coal Mine, Hebei Province",
        "Geochemistry": {
            "SiO2": "6.18%", "Al2O3": "1.40%", "Fe2O3": "1.19%"
        }
    },
    "Lean-thin coal": {
        "Origin": "Huainan Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "6.46%", "Al2O3": "0.98%", "Fe2O3": "0.44%"
        }
    },
    "Thin coal": {
        "Origin": "Pingdingshan Coal Mine, Henan Province",
        "Geochemistry": {
            "SiO2": "7.13%", "Al2O3": "1.20%", "Fe2O3": "0.54%"
        }
    },
    "Coking coal": {
        "Origin": "Liuzhi Coal Mine, Guizhou Province",
        "Geochemistry": {
            "SiO2": "3.05%", "Al2O3": "0.84%", "Fe2O3": "1.08%"
        }
    },
    "0.3 coking coal": {
        "Origin": "Xishan Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "2.48%", "Al2O3": "1.23%", "Fe2O3": "0.89%"
        }
    },
    "Fat coal": {
        "Origin": "Kailuan Coal Mine, Hebei Province",
        "Geochemistry": {
            "SiO2": "3.96%", "Al2O3": "1.00%", "Fe2O3": "1.25%"
        }
    },
    "Gas-fat coal": {
        "Origin": "Shenmu Coal Mine, Shaanxi Province",
        "Geochemistry": {
            "SiO2": "2.92%", "Al2O3": "1.68%", "Fe2O3": "0.93%"
        }
    },
    "Gas coal": {
        "Origin": "Hanjing Coal Mine, Shaanxi Province",
        "Geochemistry": {
            "SiO2": "2.05%", "Al2O3": "2.09%", "Fe2O3": "0.98%"
        }
    },
    "Lignite coal grade 1": {
        "Origin": "Baotou Coal Mine, Inner Mongolia",
        "Geochemistry": {
            "SiO2": "17.20%", "Al2O3": "1.07%", "Fe2O3": "0.86%"
        }
    },
    "Lignite coal grade 2": {
        "Origin": "Yimin Coal Mine, Inner Mongolia",
        "Geochemistry": {
            "SiO2": "21.05%", "Al2O3": "1.75%", "Fe2O3": "1.03%"
        }
    },
    # ── GANGUE CLASSES (full XRF elemental profiles) ─────────────────────────
    "Carbonaceous shale": {
        "Origin": "Dongfeng Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "31.11%", "Al2O3": "18.19%", "Fe2O3": "1.66%",
            "K2O": "1.05%", "MgO": "0.45%", "Na2O": "0.64%",
            "CaO": "0.15%", "Substrate": "46.25%"
        }
    },
    "Black shale 1": {
        "Origin": "Xuzhou Coal Mine, Jiangsu Province",
        "Geochemistry": {
            "SiO2": "47.02%", "Al2O3": "30.21%", "Fe2O3": "3.64%",
            "K2O": "0.64%", "MgO": "0.60%", "Na2O": "1.62%",
            "CaO": "2.58%", "Substrate": "13.24%"
        }
    },
    "Black shale 2": {
        "Origin": "Datong Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "42.36%", "Al2O3": "27.83%", "Fe2O3": "2.74%",
            "K2O": "1.27%", "MgO": "0.22%", "Na2O": "0.08%",
            "CaO": "0.30%", "Substrate": "24.80%"
        }
    },
    "Arenaceous shale 1": {
        "Origin": "Xinglongzhuang Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "64.51%", "Al2O3": "15.40%", "Fe2O3": "2.77%",
            "K2O": "2.29%", "MgO": "1.73%", "Na2O": "0.12%",
            "CaO": "2.73%", "Substrate": "9.89%"
        }
    },
    "Arenaceous shale 2": {
        "Origin": "Baodian Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "64.17%", "Al2O3": "14.96%", "Fe2O3": "2.93%",
            "K2O": "1.90%", "MgO": "1.45%",
            "CaO": "2.61%", "Substrate": "11.35%"
        }
    },
    "Medium-grained sandstone": {
        "Origin": "Jining Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "79.17%", "Al2O3": "8.72%", "Fe2O3": "1.95%",
            "K2O": "1.14%", "MgO": "0.52%", "Na2O": "0.53%",
            "CaO": "0.37%", "Substrate": "7.26%"
        }
    },
    "Clay": {
        "Origin": "Yanzhou Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "40.32%", "Al2O3": "30.02%", "Fe2O3": "4.44%",
            "K2O": "13.26%", "MgO": "1.00%", "Na2O": "1.63%",
            "CaO": "3.55%", "Substrate": "5.36%"
        }
    },
    "Fine-grained sandstone": {
        "Origin": "Zibo Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "77.47%", "Al2O3": "9.04%", "Fe2O3": "1.76%",
            "K2O": "1.10%", "MgO": "1.30%", "Na2O": "0.16%",
            "CaO": "2.48%", "Substrate": "6.17%"
        }
    },
    "Siltstone 1": {
        "Origin": "Huaibei Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "56.13%", "Al2O3": "13.21%", "Fe2O3": "4.50%",
            "K2O": "2.47%", "MgO": "1.42%", "Na2O": "1.42%",
            "CaO": "7.82%", "Substrate": "12.73%"
        }
    },
    "Siltstone 2": {
        "Origin": "Huainan Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "82.06%", "Al2O3": "1.66%", "Fe2O3": "3.51%",
            "K2O": "1.37%", "MgO": "2.48%", "Na2O": "0.77%",
            "CaO": "2.69%", "Substrate": "4.87%"
        }
    },
    "Argillaceous limestone 1": {
        "Origin": "Feicheng Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "13.50%", "Al2O3": "11.22%", "Fe2O3": "3.06%",
            "K2O": "2.07%", "MgO": "16.70%",
            "CaO": "26.56%", "Substrate": "26.40%"
        }
    },
    "Argillaceous limestone 2": {
        "Origin": "Zaozhuang Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "17.28%", "Al2O3": "12.08%", "Fe2O3": "2.59%",
            "K2O": "1.89%", "MgO": "4.93%", "Na2O": "1.22%",
            "CaO": "28.90%", "Substrate": "30.56%"
        }
    },
    # ── SAFETY OVERRIDE PASSPORT (main.py sets predicted_class to this string
    # when the batch-mean verdict is fuel but at least `min_flagged_scans` scans
    # each exceeded the gangue-mass threshold). Without this entry the frontend
    # showed a blank panel on a mixed-batch REJECT. ──────────────────────────
    "Gangue Contamination Detected": {
        "Origin": "Mixed batch — one or more scans exceeded gangue safety threshold",
        "Geochemistry": {}
    },
}


def load_label_encoder(path: str):
    """Legacy helper (pickle, needs scikit-learn installed). Serving uses load_class_list()."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_class_list(path: str) -> List[str]:
    """Class names in model-output order (identical to label_encoder.pkl.classes_)."""
    with open(path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    if len(classes) != 24 or len(set(classes)) != 24:
        raise ValueError(f"class_list.json must hold 24 unique class names, got {len(classes)}.")
    return classes


# Input limits / layout constants -------------------------------------------------------------
MAX_SCANS = 2000            # hard cap (memory); larger files are rejected, never subsampled
DATASET_META_PREFIX = 5     # numeric metadata rows that precede the bands in the raw dataset layout


class SpectraFormatError(ValueError):
    """The upload cannot be interpreted as NIR spectra. Message is safe to show to the user."""


@dataclass
class IngestResult:
    tensor: torch.Tensor                 # (N, 3, 1500)
    wavelengths: List[float]             # 1500 x-axis values (1000-2499 nm)
    input_format: str                    # dataset_raw | clean_spectra | unrecognized_layout
    validated: bool                      # True only for the layout the checkpoint was validated on
    notes: List[str] = field(default_factory=list)
    dropped: dict = field(default_factory=dict)


def _snv(arr: np.ndarray) -> np.ndarray:
    """Per-sample z-score (Standard Normal Variate). Input/Output: (N, L)."""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return (arr - mean) / (std + 1e-8)


def _savgol_deriv(arr: np.ndarray, order: int) -> np.ndarray:
    """Savitzky-Golay derivative, window=11, polyorder=2 (identical to training)."""
    deriv = savgol_filter(arr, window_length=11, polyorder=2, deriv=order, axis=1)
    return deriv[:, :EXPECTED_BANDS]


def load_spectra(file_bytes: bytes, max_scans: int = MAX_SCANS) -> IngestResult:
    """
    Raw CSV bytes -> model-ready tensor, mirroring the training notebook's loader.

    Order matters (it is the training order): numeric coercion -> drop metadata/text rows ->
    orientation (distance heuristic) -> remove wavelength-axis vectors -> drop non-finite scans ->
    front-crop to 1500 -> flat-line filter -> SNV -> Savitzky-Golay d1/d2.
    Nothing is ever randomly subsampled: every scan reaches the model.
    """
    # 1) schema-agnostic ingestion ------------------------------------------------------------
    try:
        df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None, low_memory=False)
    except Exception as exc:  # EmptyDataError, ParserError, UnicodeDecodeError, ...
        raise SpectraFormatError("The upload could not be read as a CSV file.") from exc
    if df_raw.empty:
        raise SpectraFormatError("The CSV file is empty.")

    n_cols = df_raw.shape[1]
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")
    del df_raw
    valid_mask = df_num.notna().sum(axis=1) > (n_cols * 0.5)
    vals = df_num[valid_mask].to_numpy(dtype=np.float32)
    del df_num
    if vals.size == 0:
        raise SpectraFormatError("No numeric spectral data found in the uploaded file.")

    # 2) orientation: the axis closer to 1500 is the spectral axis ------------------------------
    if abs(vals.shape[0] - EXPECTED_BANDS) < abs(vals.shape[1] - EXPECTED_BANDS):
        vals = vals.T                                   # -> (N_vectors, N_bands)

    # 3) drop wavelength-axis vectors (any position) and non-finite vectors ---------------------
    # A wavelength axis is strictly increasing with every value >= 900 (reflectance never is).
    if vals.shape[1] > 1:
        with np.errstate(invalid="ignore"):
            is_axis = (vals >= 900).all(axis=1) & (np.diff(vals, axis=1) > 0).all(axis=1)
    else:
        is_axis = np.zeros(vals.shape[0], dtype=bool)
    # The raw label column has five text labels followed by a wavelength axis.
    metadata_axis = np.zeros(vals.shape[0], dtype=bool)
    if vals.shape[1] == 1505:
        metadata_axis = np.isnan(vals[:, :5]).all(axis=1) & (vals[:, 5:] >= 900).all(axis=1) & (np.diff(vals[:, 5:], axis=1) > 0).all(axis=1)
    finite = np.isfinite(vals).all(axis=1)
    if (~finite & ~is_axis & ~metadata_axis).any():
        raise SpectraFormatError("Spectral values must be finite.")
    dropped = {"wavelength_axis": int(is_axis.sum()), "non_finite": int((~finite & ~is_axis).sum())}
    vals = vals[finite & ~is_axis]
    if vals.shape[0] == 0:
        raise SpectraFormatError("No valid spectra remained after removing the wavelength axis and invalid rows.")
    if vals.shape[0] > max_scans:
        raise SpectraFormatError(
            f"File contains {vals.shape[0]} scans; the limit is {max_scans}. Split the file and upload in parts."
        )

    # 4) band axis: exact-length check + layout fingerprint ------------------------------------
    n_bands = vals.shape[1]
    if n_bands not in (1500, 1505):
        raise SpectraFormatError(f"Expected 1500 or 1505 numeric spectral bands, got {n_bands}.")
    notes: List[str] = []
    if n_bands == 1505:
        from pathlib import Path
        header_sets = json.loads((Path(__file__).parent / "assets" / "header_values.json").read_text(encoding="utf-8"))["rows"]
        for index, allowed in enumerate(header_sets):
            if not np.isin(vals[:, index], allowed).all():
                raise SpectraFormatError(f"Unrecognized metadata value in header row {index + 1}.")
        input_format, validated = "dataset_raw", True
        bands = vals[:, 5:]
        retained_bands = vals[:, 5:1500]
    else:
        input_format, validated = "clean_spectra", False
        bands = retained_bands = vals
        notes.append("Clean spectrum input is outside the validated training layout and requires review.")
    if not ((bands >= -20) & (bands <= 120)).all():
        raise SpectraFormatError("Reflectance must be finite and within -20 to 120 percent.")
    if (retained_bands.std(axis=1) <= 1e-5).any():
        raise SpectraFormatError("Spectral bands contain a flat-line scan.")
    dropped["flat_line"] = 0
    vals = vals[:, :EXPECTED_BANDS]

    # 6) physics tensor -------------------------------------------------------------------------
    s0 = _snv(vals)
    s1 = _savgol_deriv(s0, order=1)
    s2 = _savgol_deriv(s0, order=2)
    tensor = torch.from_numpy(np.stack([s0, s1, s2], axis=1).astype(np.float32))
    wavelengths = np.linspace(1000, 2499, EXPECTED_BANDS).round(2).tolist()
    return IngestResult(tensor, wavelengths, input_format, validated, notes, dropped)


def process_csv_to_tensor(file_bytes: bytes):
    """Back-compat wrapper: returns (tensor (N,3,1500), wavelengths)."""
    res = load_spectra(file_bytes)
    return res.tensor, res.wavelengths
