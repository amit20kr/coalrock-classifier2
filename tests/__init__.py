"""Prerequisite checks for tests that load the real backend at import time."""
from pathlib import Path

import pytest


CHECKPOINT = Path(__file__).resolve().parents[1] / "backend" / "models" / "pcvit_expert.pth"


def require_checkpoint(*, allow_module_level=False):
    if not CHECKPOINT.is_file():
        pytest.skip(
            "missing checkpoint: backend/models/pcvit_expert.pth",
            allow_module_level=allow_module_level,
        )
