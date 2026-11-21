"""Real-file regression using an externally supplied demo CSV and checkpoint."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from tests import require_checkpoint

DEMO = Path(__file__).resolve().parent.parent / "Demo test" / "Carbonaceous shale.csv"
if not DEMO.is_file():
    pytest.skip("missing demo CSV: Demo test/Carbonaceous shale.csv", allow_module_level=True)
require_checkpoint(allow_module_level=True)

from backend.main import app

client = TestClient(app)


def test_real_carbonaceous_shale_is_rejected_with_high_class_agreement():
    r = client.post("/predict", files={"file": (DEMO.name, DEMO.read_bytes(), "text/csv")}).json()
    assert r["input"]["format"] == "dataset_raw" and r["input"]["validated"] is True
    assert r["scan_summary"]["n_scans"] == 418                          # 419 columns - 1 label column
    assert r["verdict"] == "REJECT" and r["predicted_class"] == "Carbonaceous shale"
    assert r["scan_summary"]["n_flagged"] >= 400                        # measured: 417 of 418 at >= 0.95
