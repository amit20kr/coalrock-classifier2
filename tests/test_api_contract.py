"""API contract: ops endpoints, v1 keys preserved, v2 additions, safe errors, thresholds in force."""
import numpy as np
import pandas as pd
import io
import pytest
import torch
from fastapi.testclient import TestClient

from tests import require_checkpoint

require_checkpoint(allow_module_level=True)

import backend.main as main_module
from backend.main import app

client = TestClient(app)
V1_KEYS = {"verdict", "is_gangue", "predicted_class", "confidence", "top_5", "digital_passport", "spectral_channels"}


class _Mock(torch.nn.Module):
    def __init__(self, logits): super().__init__(); self.l = logits
    def forward(self, x):
        n = x.shape[0]; out, self.l = self.l[:n], self.l[n:]; return out


def _csv(n):
    buf = io.BytesIO(); pd.DataFrame(np.random.rand(n, 1500)).to_csv(buf, index=False, header=False); return buf.getvalue()


def test_ops_endpoints():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is True
    v = client.get("/version").json(); assert len(v["weights_sha256"]) == 64
    m = client.get("/meta").json()
    assert len(m["fuel_classes"]) == 12 and len(m["gangue_classes"]) == 12
    assert m["safety"]["min_flagged_scans"] == main_module.GANGUE_OVERRIDE_MIN_SCANS


def test_real_model_response_has_v1_and_v2_keys():
    r = client.post("/predict", files={"file": ("t.csv", _csv(5), "text/csv")}).json()
    assert V1_KEYS <= set(r)
    assert {"scan_summary", "input", "warnings", "timing_ms", "request_id"} <= set(r)
    assert r["scan_summary"]["n_scans"] == 5 and len(r["scan_summary"]["per_scan_gangue_mass"]) == 5
    assert r["input"]["validated"] is False and r["warnings"]            # clean layout is flagged


def _flagged_window_response(monkeypatch, csv_dataset_raw, n_scans, n_flagged):
    logits = torch.full((n_scans, 24), -10.0)
    logits[n_flagged:, main_module.fuel_indices[0]] = 10.0
    logits[:n_flagged, main_module.gangue_indices[0]] = 10.0
    monkeypatch.setattr(main_module, "model", _Mock(logits))

    # Full-size synthetic uploads need compact decimal encoding to fit the CSV size limit.
    payload = csv_dataset_raw(n_scans, decimals=5)
    response = client.post("/predict", files={"file": ("t.csv", payload, "text/csv")})
    assert response.status_code == 200
    result = response.json()
    assert result["input"]["validated"] is True
    assert result["scan_summary"]["n_scans"] == n_scans
    assert result["scan_summary"]["n_flagged"] == n_flagged
    return result


@pytest.mark.parametrize(
    "n_scans,n_flagged,expected",
    [(3, 1, "REVIEW"), (10, 3, "REVIEW"), (20, 1, "REVIEW"),
     (21, 1, "ACCEPT"), (418, 1, "ACCEPT"),
     (3, 2, "REJECT"), (10, 4, "REJECT (Safety Override)")],
)
def test_flagged_scan_review_cutoff(monkeypatch, csv_dataset_raw, n_scans, n_flagged, expected):
    result = _flagged_window_response(monkeypatch, csv_dataset_raw, n_scans, n_flagged)
    assert result["verdict"] == expected
    assert result["plot_scan_index"] < n_flagged
    if expected == "REVIEW":
        assert any(f"{n_flagged} scan(s)" in w and "0.95" in w and "manual review" in w
                   for w in result["warnings"])
    elif expected == "ACCEPT":
        assert any("Inspect these scans manually" in w for w in result["warnings"])
    else:
        assert not any("manual review" in w for w in result["warnings"])


def test_flagged_scan_review_cutoff_can_follow_sensor(monkeypatch, csv_dataset_raw):
    monkeypatch.setattr(main_module, "REVIEW_ANY_FLAGGED_UP_TO_SCANS", 21)
    result = _flagged_window_response(monkeypatch, csv_dataset_raw, 21, 1)
    assert result["verdict"] == "REVIEW"
    safety = client.get("/meta").json()["safety"]
    assert safety["review_any_flagged_up_to_scans"] == 21
    assert "21 scans or fewer" in safety["rule"]


def test_exactly_k_flagged_scans_reject(monkeypatch):
    k = main_module.GANGUE_OVERRIDE_MIN_SCANS
    logits = torch.full((40, 24), -10.0); logits[:, 1] = 10.0
    logits[-k:, 1] = -10.0; logits[-k:, 10] = 10.0
    monkeypatch.setattr(main_module, "model", _Mock(logits))
    r = client.post("/predict", files={"file": ("t.csv", _csv(40), "text/csv")}).json()
    assert r["verdict"] == "REJECT (Safety Override)" and r["predicted_class"] == "Gangue Contamination Detected"
    assert r["digital_passport"]["Origin"]                              # override passport exists


def test_verdict_and_class_never_disagree(monkeypatch):
    """Gangue-family verdict must report a gangue class (never 'Anthracite' + REJECT)."""
    logits = torch.full((10, 24), -10.0); logits[:, 10] = 2.0; logits[:, 1] = 2.5   # fuel class slightly higher alone
    logits[:, 7] = 2.0; logits[:, 8] = 2.0                                          # but gangue mass dominates
    monkeypatch.setattr(main_module, "model", _Mock(logits))
    r = client.post("/predict", files={"file": ("t.csv", _csv(10), "text/csv")}).json()
    assert r["is_gangue"] and r["predicted_class"] in main_module.CLASSES
    assert r["predicted_class"] not in main_module.FUEL_CLASSES


def test_unexpected_error_does_not_leak_exception_text(monkeypatch):
    def boom(_): raise RuntimeError("SECRET_INTERNAL_PATH /srv/x")
    monkeypatch.setattr(main_module, "load_spectra", boom)
    r = client.post("/predict", files={"file": ("t.csv", _csv(3), "text/csv")})
    assert r.status_code == 500 and "SECRET_INTERNAL_PATH" not in r.text and "request id" in r.json()["detail"]


def test_dataset_layout_via_api_is_validated(csv_dataset_raw):
    r = client.post("/predict", files={"file": ("t.csv", csv_dataset_raw(8), "text/csv")}).json()
    assert r["input"]["format"] == "dataset_raw" and r["input"]["validated"] is True
    assert r["scan_summary"]["n_scans"] == 8


def test_late_rock_scans_in_dataset_layout_are_not_lost(monkeypatch, csv_dataset_raw):
    """F1 regression through the real parser: 100 scans, last 4 rock, chunked inference."""
    k = main_module.GANGUE_OVERRIDE_MIN_SCANS
    logits = torch.full((100, 24), -10.0); logits[:, 1] = 10.0
    logits[-k:, 1] = -10.0; logits[-k:, 10] = 10.0
    monkeypatch.setattr(main_module, "model", _Mock(logits))
    r = client.post("/predict", files={"file": ("t.csv", csv_dataset_raw(100), "text/csv")}).json()
    assert r["is_gangue"] and r["scan_summary"]["n_scans"] == 100


def test_low_confidence_fuel_batch_goes_to_review(monkeypatch):
    logits = torch.full((10, 24), -10.0); logits[:, 1] = 1.0; logits[:, 2] = 1.0; logits[:, 4] = 1.0   # 3 fuel classes share mass
    monkeypatch.setattr(main_module, "model", _Mock(logits))
    r = client.post("/predict", files={"file": ("t.csv", _csv(10), "text/csv")}).json()
    assert r["verdict"] == "REVIEW" and r["is_gangue"] is False and r["confidence"] < main_module.REVIEW_CONFIDENCE
    assert any("human check" in w for w in r["warnings"])


def test_review_never_softens_a_reject(monkeypatch):
    logits = torch.full((10, 24), -10.0); logits[:, 10] = 1.0; logits[:, 7] = 1.0; logits[:, 8] = 1.0  # confidence ~0.33 but gangue
    monkeypatch.setattr(main_module, "model", _Mock(logits))
    r = client.post("/predict", files={"file": ("t.csv", _csv(10), "text/csv")}).json()
    assert r["verdict"] == "REJECT" and r["is_gangue"] is True


def test_unvalidated_input_can_be_capped_at_review(monkeypatch):
    monkeypatch.setattr(main_module, "REVIEW_UNVALIDATED_INPUT", True)
    logits = torch.full((10, 24), -10.0); logits[:, 1] = 10.0
    monkeypatch.setattr(main_module, "model", _Mock(logits))
    r = client.post("/predict", files={"file": ("t.csv", _csv(10), "text/csv")}).json()   # clean layout => unvalidated
    assert r["verdict"] == "REVIEW" and r["input"]["validated"] is False
