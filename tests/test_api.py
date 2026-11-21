import pytest
from fastapi.testclient import TestClient

from tests import require_checkpoint

require_checkpoint(allow_module_level=True)

from backend.main import app
import pandas as pd
import numpy as np
import io

client = TestClient(app)

def test_api_oversize():
    # 11 MB of zeros
    content = b"0" * (11 * 1024 * 1024)
    response = client.post("/predict", files={"file": ("large.csv", content, "text/csv")})
    assert response.status_code == 413
    assert "Payload Too Large" in response.json()["detail"]

def test_api_empty_file():
    response = client.post("/predict", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 422
    # pandas raises EmptyDataError or similar which we catch and raise as 422
    # check that we don't return 500
    assert "detail" in response.json()

def test_api_malformed_csv():
    content = b"this,is,not,a,valid,spectra,file"
    response = client.post("/predict", files={"file": ("bad.csv", content, "text/csv")})
    assert response.status_code == 422

def test_api_binary_garbage():
    content = b"\x00\x01\x02\x03\x04" * 100
    response = client.post("/predict", files={"file": ("garbage.csv", content, "text/csv")})
    assert response.status_code == 422

def test_api_wavelength_axis_as_column():
    # Spectra are usually (N_samples, 1500) where rows are samples.
    # What if it's (1500, N_samples)? The pipeline transposes it automatically.
    # Let's verify it works without throwing an error.
    # Generate 5 valid spectra of 1500 bands, transposed to (1500, 5)
    data = np.column_stack([np.arange(1000, 2500), np.random.default_rng(42).random((1500, 5))])
    df = pd.DataFrame(data)
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False, header=False)
    
    response = client.post("/predict", files={"file": ("transposed.csv", csv_buf.getvalue(), "text/csv")})
    assert response.status_code == 200
    res = response.json()
    assert res["scan_summary"]["n_scans"] == 5
    assert res["input"]["dropped"]["wavelength_axis"] == 1
