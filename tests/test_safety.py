import pytest
import torch
import numpy as np
import pandas as pd
import io
from fastapi.testclient import TestClient

from tests import require_checkpoint

require_checkpoint(allow_module_level=True)

from backend.main import app
import backend.main as main_module

client = TestClient(app)

@pytest.fixture
def mock_model(monkeypatch):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mock_logits = None
        def forward(self, x):
            if self.mock_logits is not None:
                # return the next chunk of mock logits
                N = x.shape[0]
                logits = self.mock_logits[:N]
                self.mock_logits = self.mock_logits[N:]
                return logits
            return torch.zeros(x.shape[0], 24)
            
    model = MockModel()
    monkeypatch.setattr(main_module, "model", model)
    return model

def create_csv(n_scans=10):
    data = np.random.rand(n_scans, 1500)
    df = pd.DataFrame(data)
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False, header=False)
    return csv_buf.getvalue()

def test_pure_fuel_accept(mock_model, csv_dataset_raw):
    csv_data = csv_dataset_raw(10)
    # Fuel class index (e.g., 1 for Anthracite coal grade 1)
    # We set logits such that fuel class is very high
    logits = torch.ones(10, 24) * -10.0
    logits[:, 1] = 10.0
    mock_model.mock_logits = logits
    
    response = client.post("/predict", files={"file": ("test.csv", csv_data, "text/csv")})
    assert response.status_code == 200
    res = response.json()
    assert res["verdict"] == "ACCEPT"
    assert res["is_gangue"] is False

def test_pure_gangue_reject(mock_model):
    csv_data = create_csv(10)
    # Gangue class index (e.g., 10 for Clay)
    logits = torch.ones(10, 24) * -10.0
    logits[:, 10] = 10.0
    mock_model.mock_logits = logits
    
    response = client.post("/predict", files={"file": ("test.csv", csv_data, "text/csv")})
    assert response.status_code == 200
    res = response.json()
    assert res["verdict"].startswith("REJECT")
    assert res["is_gangue"] is True

def test_mixed_batch_reject_override(mock_model):
    csv_data = create_csv(10)
    # 6 fuel, 4 gangue. Mean is fuel, but we meet the >= 4 gangue scans override
    logits = torch.ones(10, 24) * -10.0
    logits[:6, 1] = 10.0 # Fuel
    logits[6:, 10] = 10.0 # Gangue
    mock_model.mock_logits = logits
    
    response = client.post("/predict", files={"file": ("test.csv", csv_data, "text/csv")})
    res = response.json()
    assert res["verdict"].startswith("REJECT")
    assert res["is_gangue"] is True

    assert res["verdict"] == "REJECT (Safety Override)"
    assert res["scan_summary"]["n_flagged"] == 4
    assert res["scan_summary"]["mean_gangue_mass"] < 0.5

def test_split_gangue_mass_flagged(mock_model):
    # F4 Regression: Scan split 0.5/0.4 across two rock classes
    csv_data = create_csv(4)
    logits = torch.ones(4, 24) * -10.0
    # Make them all fuel initially
    logits[:, 1] = 10.0
    
    # But make one scan have high total gangue mass, split across two classes
    # e.g., Clay (10) and Siltstone (21)
    # log(prob) needs to be such that softmax output gives 0.5 and 0.48
    # We can just set the values directly if we patch softmax, but it's easier
    # to set logits such that they dominate.
    # Actually, we need 4 scans to flag it (GANGUE_OVERRIDE_MIN_SCANS = 4).
    # Let's just make all 4 scans have split gangue mass.
    logits[:, 10] = 0.5
    logits[:, 21] = 0.5
    logits[:, 1] = -10.0
    # Softmax of [0.5, 0.5] vs others at -10 is ~0.5 each.
    mock_model.mock_logits = logits
    
    response = client.post("/predict", files={"file": ("test.csv", csv_data, "text/csv")})
    res = response.json()
    assert res["verdict"].startswith("REJECT")
    assert res["is_gangue"] is True

    assert res["scan_summary"]["n_flagged"] == 4
    assert min(res["scan_summary"]["per_scan_gangue_mass"]) >= 0.95

def test_late_rock_scan_not_lost(mock_model):
    # F1 Regression: Rock scan positioned late in a large file
    # We send 100 scans processed in chunks of 8.
    # The last 4 scans are pure gangue and must not be lost.
    csv_data = create_csv(100)
    logits = torch.ones(100, 24) * -10.0
    logits[:, 1] = 10.0          # All fuel initially
    logits[-4:, 1] = -10.0       # Clear fuel for last 4
    logits[-4:, 10] = 10.0       # Last 4 are gangue (Clay)
    mock_model.mock_logits = logits

    response = client.post("/predict", files={"file": ("test.csv", csv_data, "text/csv")})
    res = response.json()
    assert res["verdict"].startswith("REJECT")
    assert res["is_gangue"] is True

    assert res["verdict"] == "REJECT (Safety Override)"
    assert res["scan_summary"]["n_flagged"] == 4
    assert res["scan_summary"]["mean_gangue_mass"] < 0.5

def test_review_never_downgrades_reject(mock_model):
    csv_data = create_csv(4)
    # Gangue override triggers (4 scans)
    logits = torch.ones(4, 24) * -10.0
    logits[:, 10] = 10.0
    # But batch mean confidence is artificially very low (this is hard to simulate without
    # making gangue override fail, but wait, batch mean confidence is top5_probs[0]).
    # If all 24 classes are equal, confidence is 1/24. 
    # Let's make all scans perfectly uniform, so gangue mass = 12/24 = 0.5. But threshold is 0.95.
    # We need gangue mass > 0.95.
    # So 2 gangue classes share 1.0 probability (0.5 each). Then top class confidence is 0.5.
    # 0.5 < 0.60 (REVIEW threshold).
    logits[:, 10] = 0.0
    logits[:, 7] = 0.0
    # others are -10.0
    # softmax gives ~0.5 and ~0.5. Gangue mass is ~1.0. Flagged!
    # batch mean top class confidence is ~0.5. REVIEW!
    # Since flagged >= 4, it should be REJECT, not REVIEW.
    mock_model.mock_logits = logits
    
    response = client.post("/predict", files={"file": ("test.csv", csv_data, "text/csv")})
    res = response.json()
    assert res["verdict"].startswith("REJECT")

    assert res["confidence"] < main_module.REVIEW_CONFIDENCE
    assert res["scan_summary"]["n_flagged"] == 4
