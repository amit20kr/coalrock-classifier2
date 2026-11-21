"""External raw corpus, supplied by SPECTRASENSE_RAW_DIR."""
import io
import json
import os
from pathlib import Path
import pandas as pd
import pytest
from backend.pipeline import load_spectra

ROOT = Path(__file__).resolve().parents[1]
CLASSES = json.loads((ROOT / "backend/assets/class_list.json").read_text(encoding="utf-8"))

@pytest.mark.parametrize("name", CLASSES)
def test_every_raw_scan_passes_validation(name):
    directory = os.environ.get("SPECTRASENSE_RAW_DIR")
    if not directory:
        pytest.skip("Set SPECTRASENSE_RAW_DIR to run the external raw corpus.")
    path = Path(directory) / (name + ".csv")
    raw = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)
    result = load_spectra(raw)
    assert result.validated is True
    assert result.tensor.shape == (frame.shape[1] - 1, 3, 1500)
