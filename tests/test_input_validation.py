import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.pipeline import load_spectra, SpectraFormatError
from tests import require_checkpoint

def csv(values):
    return pd.DataFrame(values).to_csv(index=False, header=False).encode()

@pytest.mark.parametrize("length", [1499, 1501, 1504, 1506, 1600])
def test_refuse_unexpected_lengths(length):
    with pytest.raises(SpectraFormatError):
        load_spectra(csv(np.linspace(1, 20, length)[None, :]))

def test_refuse_unobserved_header_value(csv_dataset_raw):
    frame = pd.read_csv(io.BytesIO(csv_dataset_raw()), header=None)
    frame.iloc[0, 1:] = 19
    with pytest.raises(SpectraFormatError):
        load_spectra(csv(frame))

@pytest.mark.parametrize("value", [-20.001, 120.001, float("inf"), float("nan")])
def test_refuse_invalid_trailing_values(csv_dataset_raw, value):
    frame = pd.read_csv(io.BytesIO(csv_dataset_raw()), header=None)
    frame.iloc[-1, 1] = value
    with pytest.raises(SpectraFormatError):
        load_spectra(csv(frame))

def test_flat_bands_cannot_hide_behind_metadata(csv_dataset_raw):
    frame = pd.read_csv(io.BytesIO(csv_dataset_raw()), header=None)
    frame.iloc[7:, 1:] = 12
    with pytest.raises(SpectraFormatError):
        load_spectra(csv(frame))

def test_trailing_five_leave_tensor_status_and_verdict_unchanged(csv_dataset_raw):
    require_checkpoint()
    from backend.main import app

    original = csv_dataset_raw()
    frame = pd.read_csv(io.BytesIO(original), header=None)
    frame.iloc[-5:, 1:] = 42
    changed = csv(frame)
    before, after = load_spectra(original), load_spectra(changed)
    assert before.validated is after.validated is True
    np.testing.assert_array_equal(before.tensor.numpy(), after.tensor.numpy())
    client = TestClient(app)
    first = client.post("/predict", files={"file": ("a.csv", original)}).json()
    second = client.post("/predict", files={"file": ("b.csv", changed)}).json()
    assert first["verdict"] == second["verdict"]
    assert first["scan_summary"] == second["scan_summary"]

@pytest.mark.parametrize("value", [-20, -5.325, 102.856, 120])
def test_design_margin_values_are_accepted(value):
    values = np.linspace(1, 20, 1500)[None, :]
    values[0, 100] = value
    assert load_spectra(csv(values)).tensor.shape == (1, 3, 1500)

@pytest.mark.parametrize("value", [-20.001, 120.001, float("nan"), float("inf")])
def test_invalid_retained_values_are_refused(value):
    values = np.tile(np.linspace(1, 20, 1500), (3, 1))
    values[0, 100] = value
    with pytest.raises(SpectraFormatError):
        load_spectra(csv(values))
