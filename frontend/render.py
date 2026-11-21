"""
frontend/render.py — presentation layer for the Spectra-Sense dashboard.

Pure display code: takes the /predict JSON and draws it. Kept separate from app.py so it can be tested
without a running backend (see tests/test_frontend_render.py).

Claims policy: every number shown comes from docs/CLAIMS_LEDGER.md or from the API response itself.
The shaded spectral windows are literature reference regions, NOT detections.
"""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Fallback only: the live list is read from the backend's /meta endpoint (single source of truth).
DEFAULT_FUEL_CLASSES = [
    "Anthracite coal grade 1", "Anthracite coal grade 2",
    "Lean coal", "Lean-thin coal", "Thin coal", "Coking coal",
    "0.3 coking coal", "Fat coal", "Gas-fat coal", "Gas coal",
    "Lignite coal grade 1", "Lignite coal grade 2",
]

BG, GRID, TXT = "#0e1117", "#2c3e50", "#ecf0f1"


def _valid_response(data):
    """Validate fields consumed by the dashboard before drawing any panels."""
    def number(value):
        return type(value) in (int, float) and np.isfinite(value)

    try:
        assert isinstance(data, dict)
        assert data["verdict"] in ("ACCEPT", "REVIEW", "REJECT", "REJECT (Safety Override)")
        assert type(data["is_gangue"]) is bool
        assert isinstance(data["predicted_class"], str) and data["predicted_class"]
        assert number(data["confidence"]) and 0 <= data["confidence"] <= 1
        summary = data["scan_summary"]
        for key in ("n_scans", "n_flagged", "min_flagged_to_reject"):
            assert type(summary[key]) is int
        assert summary["n_scans"] > 0 and 0 <= summary["n_flagged"] <= summary["n_scans"]
        for key in ("flag_threshold", "mean_gangue_mass", "max_gangue_mass"):
            assert number(summary[key]) and 0 <= summary[key] <= 1
        masses = summary["per_scan_gangue_mass"]
        assert isinstance(masses, list) and len(masses) == summary["n_scans"]
        assert all(number(x) and 0 <= x <= 1 for x in masses)
        spectrum = data["spectral_channels"]
        for key in ("wavelengths", "snv", "second_derivative"):
            assert isinstance(spectrum[key], list) and len(spectrum[key]) == 1500
            assert all(number(x) for x in spectrum[key])
        assert isinstance(data["digital_passport"]["Origin"], str)
        assert isinstance(data["digital_passport"]["Geochemistry"], dict)
        assert isinstance(data["top_5"], list) and data["top_5"]
        for item in data["top_5"]:
            assert isinstance(item["class"], str)
            assert number(item["probability"]) and 0 <= item["probability"] <= 1
        assert isinstance(data["warnings"], list)
        assert all(isinstance(w, str) for w in data["warnings"])
        assert isinstance(data["input"]["format"], str)
        assert type(data["input"]["validated"]) is bool
        assert isinstance(data["request_id"], str)
        assert type(data["plot_scan_index"]) is int
        assert 0 <= data["plot_scan_index"] < summary["n_scans"]
    except (AssertionError, KeyError, TypeError, ValueError):
        return False
    return True


def _verdict_banner(data):
    valid = _valid_response(data)
    verdict = data.get("verdict") if isinstance(data, dict) else None
    if not valid:
        if verdict in ("REJECT", "REJECT (Safety Override)"):
            st.error("REJECT — response incomplete. Divert as gangue.")
        elif verdict == "REVIEW":
            st.warning("REVIEW — response incomplete. Manual check required.")
        else:
            st.error("Invalid prediction response. Manual check required.")
        return False
    if verdict == "ACCEPT" and data["is_gangue"] is not False:
        st.error("Inconsistent prediction response. Manual check required.")
        return False
    verdict = data.get("verdict", "UNKNOWN")
    pred = data.get("predicted_class", "Unknown")
    conf = data.get("confidence", 0.0)
    s = data.get("scan_summary", {})
    n, flagged = s.get("n_scans", 0), s.get("n_flagged", 0)
    thr, k = s.get("flag_threshold", 0.95), s.get("min_flagged_to_reject", 4)

    if verdict.startswith("REJECT (Safety Override)"):
        st.error("### 🚫 BATCH CONTAMINATED — SAFETY OVERRIDE")
        st.markdown(
            f"**{flagged} of {n} scans** each have ≥ {thr:.2f} total gangue probability "
            f"(≥ {k} required). The batch average alone would have passed as coal. "
            f"| Verdict: `{verdict}`"
        )
    elif verdict == "REJECT":
        st.error("### 🚫 PRIMARY CLASS: ROCK (Gangue/Waste)")
        st.markdown(
            f"**Detected:** `{pred}` — **WASTE ROCK. Conveyor flagged.** "
            f"| Mean top-class probability: **{conf * 100:.2f}%** | Verdict: `{verdict}`"
        )
    elif verdict == "REVIEW":
        st.warning("### ⚠️ REVIEW — human check recommended")
        st.markdown(
            f"**Leading class:** `{pred}` | Mean top-class probability: **{conf * 100:.2f}%** "
            f"| Verdict: `{verdict}`"
        )
    elif verdict == "ACCEPT":
        st.success("### ✅ PRIMARY CLASS: COAL (Fuel-Grade)")
        st.markdown(
            f"**Detected:** `{pred}` — **Fuel-grade coal. Accept.** "
            f"| Mean top-class probability: **{conf * 100:.2f}%** | Verdict: `{verdict}`"
        )

    action = {"ACCEPT": "Send to crusher", "REVIEW": "Manual check",
              "REJECT": "Divert as gangue", "REJECT (Safety Override)": "Divert as gangue"}[verdict]
    st.markdown(f"**Action — {action}**")

    for w in data.get("warnings", []):
        st.warning(w, icon="⚠️")

    inp = data.get("input", {})
    if inp:
        label = "validated layout" if inp.get("validated") else "UNVALIDATED layout"
        st.caption(f"Input: `{inp.get('format', '?')}` ({label}) · request `{data.get('request_id', '-')}`")

    return True


def _scan_metrics(data):
    s = data.get("scan_summary")
    if not s:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scans analysed", f"{s['n_scans']}")
    c2.metric("Scans flagged as gangue", f"{s['n_flagged']}", help=f"Total gangue probability ≥ {s['flag_threshold']:.2f}")
    c3.metric("Mean gangue probability", f"{s['mean_gangue_mass'] * 100:.1f}%")
    c4.metric("Max gangue probability", f"{s['max_gangue_mass'] * 100:.1f}%")


def _spectral_plot(data):
    st.markdown("### 📈 Spectral channels")
    sp = data.get("spectral_channels", {})
    wl, snv, d2 = sp.get("wavelengths", []), sp.get("snv", []), sp.get("second_derivative", [])
    if not (wl and snv):
        st.warning("No spectral data returned from backend.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    wl = np.array(wl)
    ax.plot(wl, np.array(snv), color="#3498db", linewidth=1.4, zorder=3)
    if d2:
        twin = ax.twinx()
        twin.plot(wl, d2, color="#e74c3c", linewidth=0.8, alpha=0.6)
        twin.set_ylabel("2nd Derivative", color="#e74c3c", fontsize=9)
        twin.tick_params(axis="y", colors="#e74c3c")
        twin.set_facecolor(BG)

    y_min, y_max = ax.get_ylim(); h = y_max - y_min
    for lo, hi, color, name in ((1350, 1450, "#6c9fdb", "H₂O region\n(~1400 nm)"),
                                (1850, 1950, "#95a5a6", "H₂O region\n(~1900 nm)"),
                                (2150, 2250, "#e74c3c", "Al-OH region\n(~2200 nm)")):
        ax.axvspan(lo, hi, color=color, alpha=0.14, zorder=1)
        ax.text((lo + hi) / 2, y_max - 0.04 * h, name, color=color, fontsize=7, ha="center", va="top",
                fontweight="bold", style="italic")

    idx, n = data.get("plot_scan_index", 0), data.get("scan_summary", {}).get("n_scans", 1)
    ax.set_xlabel("Wavelength (nm)", color=TXT, fontsize=10)
    ax.set_ylabel("SNV-normalised reflectance", color="#3498db", fontsize=10)
    ax.set_title(f"Scan {idx + 1} of {n}", color=TXT, pad=12, fontsize=11, fontweight="bold")
    ax.tick_params(colors=TXT); ax.spines[:].set_color(GRID)
    ax.legend(handles=[mpatches.Patch(color="#3498db", label="SNV channel"),
                       mpatches.Patch(color="#e74c3c", alpha=0.6, label="2nd derivative")],
              loc="lower right", facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TXT, fontsize=7)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Shaded bands mark literature absorption regions for orientation only — they are not detections. "
               "The most gangue-like scan is plotted when any scan is flagged, otherwise scan 1.")


def _passport(data, images_dir):
    st.markdown("### 🔬 Digital Passport")
    pred = data.get("predicted_class", "Unknown")
    p = data.get("digital_passport", {})
    path = os.path.join(images_dir, f"{pred}.jpg")
    if os.path.exists(path):
        st.image(path, caption=pred, use_container_width=True)
    else:
        st.info("No visual reference available for this class.")
    st.markdown(
        f"| Property | Value |\n|---|---|\n"
        f"| **Origin** | {p.get('Origin', 'Standard Mining Library')} |\n"
        f"| **Class** | {pred} |\n"
        f"| **Mean top-class probability** | {data.get('confidence', 0.0) * 100:.2f}% |\n"
        f"| **Verdict** | `{data.get('verdict', '?')}` |\n"
    )


def _top5(data, fuel_classes):
    st.markdown("### 📊 Mean class probability (Top 5)")
    top5 = data.get("top_5", [])
    if not top5:
        return
    classes = [i["class"] for i in reversed(top5)]
    probs = [i["probability"] * 100 for i in reversed(top5)]
    colors = ["#2ecc71" if c in fuel_classes else "#e74c3c" for c in classes]
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(classes, probs, color=colors, height=0.6, edgecolor="none")
    for bar, prob in zip(bars, probs):
        inside = prob > 50
        ax.text(bar.get_width() + (-1.5 if inside else 0.5), bar.get_y() + bar.get_height() / 2, f"{prob:.2f}%",
                va="center", ha="right" if inside else "left", color="white" if inside else TXT,
                fontsize=8, fontweight="bold")
    ax.set_xlim(0, 105); ax.set_xlabel("Probability (%)", color=TXT, fontsize=9)
    ax.tick_params(axis="y", colors=TXT, labelsize=8); ax.tick_params(axis="x", colors=TXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.legend(handles=[mpatches.Patch(color="#2ecc71", label="Fuel (Coal)"), mpatches.Patch(color="#e74c3c", label="Gangue (Rock)")],
              loc="lower right", facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TXT, fontsize=7)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def _geochemistry(data):
    st.markdown("### 🧪 Geochemistry (XRF reference)")
    geo = data.get("digital_passport", {}).get("Geochemistry", {})
    if not geo:
        st.markdown("*No reference geochemistry for this result.*")
        return
    cols = st.columns(3)
    for i, (el, pct) in enumerate(geo.items()):
        with cols[i % 3]:
            st.metric(label=el, value=pct)
    st.caption("Class-level reference values from the dataset's XRF measurements, not a measurement of this sample.")


def _scan_strip(data):
    s = data.get("scan_summary", {})
    mass = s.get("per_scan_gangue_mass", [])
    if len(mass) < 2:
        return
    st.markdown("### 🧭 Per-scan gangue probability")
    thr = s.get("flag_threshold", 0.95)
    fig, ax = plt.subplots(figsize=(10, 2.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.bar(range(1, len(mass) + 1), mass, width=1.0,
           color=["#e74c3c" if m >= thr else "#2ecc71" for m in mass], edgecolor="none")
    ax.axhline(thr, color="#f1c40f", linewidth=1, linestyle="--")
    ax.text(len(mass), thr, f" flag threshold {thr:.2f}", color="#f1c40f", fontsize=7, va="bottom", ha="right")
    ax.set_ylim(0, 1.05); ax.set_xlim(0.5, len(mass) + 0.5)
    ax.set_xlabel("Scan index (file order)", color=TXT, fontsize=9); ax.set_ylabel("P(gangue)", color=TXT, fontsize=9)
    ax.tick_params(colors=TXT, labelsize=8); ax.spines[:].set_color(GRID)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def render_result(data, fuel_classes, images_dir):
    """Draw the full result dashboard for one /predict response."""
    st.info(
        "Scope and limits — Decision-support demo using a lab spectral library. "
        "24 classes, 1000–2500 nm, 1500 bands. Held-out accuracy 95.72% on 1,566 test scans "
        "with one rock scan called coal. Unvalidated input goes to REVIEW. "
        "Not validated under underground field conditions."
    )
    if not _verdict_banner(data):
        return
    _scan_metrics(data)
    st.markdown("---")
    col_plot, col_img = st.columns([3, 2], gap="large")
    with col_plot:
        _spectral_plot(data)
    with col_img:
        _passport(data, images_dir)
    st.markdown("---")
    _scan_strip(data)
    col_bars, col_geo = st.columns([3, 2], gap="large")
    with col_bars:
        _top5(data, fuel_classes)
    with col_geo:
        _geochemistry(data)
