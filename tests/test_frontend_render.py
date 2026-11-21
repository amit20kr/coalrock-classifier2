"""Frontend tests (Streamlit AppTest): banners per verdict, no fabricated detections, no banned claims."""
import json
import sys
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1", reason="missing frontend prerequisite: streamlit.testing.v1")
AppTest = st_testing.AppTest

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
BANNED = ["0.00%", "real-time", "Real-time", "<20", "< 20", "Pyramid", "Point Cloud", "guarantee"]


def _base(verdict="ACCEPT", is_gangue=False, cls="Coking coal", n=10, flagged=0, warnings=None):
    wl = [1000 + i for i in range(1500)]
    return {
        "verdict": verdict, "is_gangue": is_gangue, "predicted_class": cls, "confidence": 0.9,
        "top_5": [{"class": cls, "probability": 0.9}, {"class": "Clay", "probability": 0.05}],
        "digital_passport": {"Origin": "Test Mine", "Geochemistry": {"SiO2": "3.05%"}},
        "spectral_channels": {"wavelengths": wl, "snv": [0.0] * 1500, "second_derivative": [0.0] * 1500},
        "request_id": "abc123", "plot_scan_index": 0,
        "scan_summary": {"n_scans": n, "n_flagged": flagged, "flag_threshold": 0.95, "min_flagged_to_reject": 4,
                         "mean_gangue_mass": 0.1, "max_gangue_mass": 0.2, "per_scan_gangue_mass": [0.1] * n},
        "input": {"format": "dataset_raw", "validated": True, "dropped": {}},
        "warnings": warnings or [],
    }


def _run(data):
    def script(payload_json, frontend_dir):
        import json, sys
        sys.path.insert(0, frontend_dir)
        from render import render_result, DEFAULT_FUEL_CLASSES
        render_result(json.loads(payload_json), DEFAULT_FUEL_CLASSES, "/nonexistent")
    at = AppTest.from_function(script, args=(json.dumps(data), str(FRONTEND)), default_timeout=30)
    return at.run()


def _text(at):
    parts = [e.value for e in at.markdown] + [e.value for e in at.error] + [e.value for e in at.warning] \
          + [e.value for e in at.success] + [e.value for e in at.caption] + [e.value for e in at.info]
    return "\n".join(map(str, parts))


def test_accept_banner():
    at = _run(_base())
    assert not at.exception
    assert any("COAL" in str(e.value) for e in at.success)


def test_reject_banner():
    at = _run(_base("REJECT", True, "Clay", flagged=10))
    assert not at.exception and any("ROCK" in str(e.value) for e in at.error)


def test_override_banner_mentions_counts():
    at = _run(_base("REJECT (Safety Override)", True, "Gangue Contamination Detected", n=40, flagged=5))
    assert not at.exception
    assert any("SAFETY OVERRIDE" in str(e.value) for e in at.error)
    assert "5 of 40 scans" in _text(at)


def test_review_banner_and_warnings_render():
    at = _run(_base("REVIEW", False, warnings=["Mean top-class confidence 0.50 is below 0.60"]))
    assert not at.exception
    assert any("REVIEW" in str(e.value) for e in at.warning)
    assert "below 0.60" in _text(at)


def test_unvalidated_layout_is_labelled():
    d = _base(); d["input"] = {"format": "clean_spectra", "validated": False, "dropped": {}}
    assert "UNVALIDATED" in _text(_run(d))


def test_no_fabricated_detection_text():
    txt = _text(_run(_base()))
    assert "DETECTED FEATURES" not in txt and "not detections" in txt


def test_app_smoke_without_upload_and_no_banned_claims():
    sys.path.insert(0, str(FRONTEND))
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    assert not at.exception
    txt = _text(at)
    for b in BANNED:
        assert b not in txt, f"banned string in UI: {b}"
    assert "95.72%" in txt and "1 rock→coal boundary error in 1,566" in txt


def test_banned_strings_absent_from_frontend_source():
    src = "\n".join(p.read_text(encoding="utf-8") for p in FRONTEND.glob("*.py"))
    for b in BANNED:
        assert b not in src, f"banned string in frontend source: {b}"


@pytest.mark.parametrize("payload", [
    {}, {"verdict": "REJECT"}, {"verdict": "REVIEW"}, {"verdict": None},
    {**_base(), "is_gangue": 0}, {**_base(), "confidence": "high"},
    {**_base(), "verdict": "UNKNOWN"}, {**_base(), "scan_summary": None},
])
def test_invalid_response_never_green_or_crashes(payload):
    at = _run(payload)
    assert not at.exception
    assert not at.success
    assert at.error or at.warning or at.info


@pytest.mark.parametrize("verdict, gangue, action", [
    ("ACCEPT", False, "Send to crusher"), ("REVIEW", False, "Manual check"),
    ("REJECT", True, "Divert as gangue")])
def test_operator_action_and_scope(verdict, gangue, action):
    at = _run(_base(verdict, gangue))
    assert not at.exception
    text = _text(at)
    assert action in text
    assert text.count("Scope and limits") == 1
    for phrase in ("lab spectral library", "24 classes", "1000–2500 nm", "1500 bands",
                   "95.72%", "1,566", "one rock scan called coal", "underground field conditions"):
        assert phrase in text


def test_mine_applications_panel_renders():
    sys.path.insert(0, str(FRONTEND))
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert any(e.label == "How this is used in a mine" for e in at.expander)
    for phrase in ("Sample check", "Belt-segment screening", "Lithology and grade passport",
                   "not yet validated under underground conditions"):
        assert phrase in _text(at)
