"""Checkpoint / architecture tests against the real weights."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from backend.model import PCViT

BASE = Path(__file__).resolve().parent.parent / "backend"
CKPT = BASE / "models" / "pcvit_expert.pth"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="missing checkpoint: backend/models/pcvit_expert.pth")


@pytest.fixture(scope="module")
def model():
    m = PCViT(24)
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    return m.eval()


def test_checkpoint_key_names():
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert {k.split(".")[0] for k in sd} == {"stem", "cls", "pos", "tf", "head"}


def test_param_count_matches_ledger(model):
    assert sum(p.numel() for p in model.parameters()) == 903_576


def test_output_shape(model):
    with torch.inference_mode():
        assert model(torch.randn(5, 3, 1500)).shape == (5, 24)


def test_eval_determinism(model):
    x = torch.randn(4, 3, 1500)
    with torch.inference_mode():
        assert torch.equal(model(x), model(x))


def test_batch_invariance(model):
    x = torch.randn(8, 3, 1500)
    with torch.inference_mode():
        batch = model(x)
        singles = torch.cat([model(x[i:i + 1]) for i in range(8)])
    assert torch.allclose(batch, singles, atol=1e-4)


def test_class_list_matches_label_encoder_order():
    classes = json.loads((BASE / "assets" / "class_list.json").read_text(encoding="utf-8"))
    assert classes == sorted(classes) and len(classes) == 24      # LabelEncoder order is alphabetical
