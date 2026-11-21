"""Shared fixtures: synthetic spectra and builders for every CSV layout the pipeline must handle."""
import io
import numpy as np
import pandas as pd
import pytest


def _spectra(n, seed=0):
    """Smooth, positive, reflectance-like spectra (N, 1500) with per-scan variation."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, 1500)
    out = []
    for _ in range(n):
        base = 10 + 15 * rng.rand() + 8 * np.sin(2 * np.pi * x * (1 + rng.rand())) + 5 * x
        out.append(base + rng.randn(1500) * 0.05)
    return np.asarray(out, dtype=np.float32)


def _to_csv(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False, header=False)
    return buf.getvalue()


@pytest.fixture
def make_spectra():
    return _spectra


@pytest.fixture
def csv_row_major():
    """(N, 1500): one scan per row, no axis, no metadata (clean instrument export)."""
    return lambda n=6: _to_csv(pd.DataFrame(_spectra(n)))


@pytest.fixture
def csv_row_major_with_axis():
    """Wavelength axis stored as the FIRST ROW."""
    return lambda n=6: _to_csv(pd.DataFrame(np.vstack([np.arange(1000, 2500, dtype=np.float32), _spectra(n)])))


@pytest.fixture
def csv_col_major_with_axis():
    """(1500, N+1): wavelength axis stored as the FIRST COLUMN, no metadata rows."""
    return lambda n=6: _to_csv(pd.DataFrame(np.hstack([np.arange(1000, 2500, dtype=np.float32)[:, None], _spectra(n).T])))


@pytest.fixture
def csv_dataset_raw():
    """The real dataset layout: 5 numeric metadata rows, granularity text row, 'Wavelength/nm' row,
    1500 band rows; column 0 holds labels / wavelengths; one column per scan."""
    def build(n=6, angle=20, *, decimals=None):
        spectra = _spectra(n).T
        if decimals is not None:
            spectra = spectra.astype(np.float64).round(decimals)                                       # (1500, n)
        lab = ["Light source angles/Degree", "Light source distance/dm", "Detection angles/Degree",
               "Detection distance/dm", "Detection azimuth angles/Degree"]
        meta_vals = [[angle] * n, [5] * n, [0] * n, [5] * n, [0] * n]
        rows = [[lab[i]] + meta_vals[i] for i in range(5)]
        rows.append(["Granularity/mm"] + ["Block with smooth surface"] * n)
        rows.append(["Wavelength/nm"] + [""] * n)
        for b in range(1500):
            rows.append([1000 + b] + spectra[b].tolist())
        return _to_csv(pd.DataFrame(rows))
    return build
