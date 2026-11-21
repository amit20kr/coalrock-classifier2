"""Pipeline unit tests: signal invariants, orientation, wavelength-axis handling, error paths."""
import io
import numpy as np
import pandas as pd
import pytest
from backend.pipeline import (SpectraFormatError, load_spectra, _snv, _savgol_deriv, MAX_SCANS)


def test_snv_zero_mean_unit_variance(make_spectra):
    z = _snv(make_spectra(8))
    np.testing.assert_allclose(z.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=1), 1.0, atol=1e-4)


def test_snv_scale_and_offset_invariance(make_spectra):
    x = make_spectra(5).astype(np.float64)
    np.testing.assert_allclose(_snv(x), _snv(3.7 * x + 11.0), atol=1e-5)


def test_second_derivative_peaks_at_absorption_dip():
    x = np.zeros((1, 1500)); idx = 1200
    x[0] = 50 - 10 * np.exp(-0.5 * ((np.arange(1500) - idx) / 8.0) ** 2)   # a reflectance DIP at band 1200
    d2 = _savgol_deriv(_snv(x), 2)[0]
    assert abs(int(np.argmax(d2)) - idx) <= 2          # dip in reflectance -> positive 2nd-derivative peak


def test_row_major_clean(csv_row_major):
    r = load_spectra(csv_row_major(6))
    assert r.tensor.shape == (6, 3, 1500) and r.input_format == "clean_spectra" and r.validated is False
    assert r.notes, "clean layout must carry an 'unvalidated' note"


def test_orientation_detection_col_major(make_spectra):
    csv = pd.DataFrame(make_spectra(7).T)                 # (1500, 7): bands as rows
    buf = io.BytesIO(); csv.to_csv(buf, index=False, header=False)
    assert load_spectra(buf.getvalue()).tensor.shape == (7, 3, 1500)


def test_wavelength_axis_as_row_is_dropped(csv_row_major_with_axis):
    assert load_spectra(csv_row_major_with_axis(6)).tensor.shape[0] == 6


def test_wavelength_axis_as_column_is_dropped_not_scored(csv_col_major_with_axis):
    """Regression: a wavelength COLUMN used to survive and be scored as an extra 'scan' (a linear ramp)."""
    r = load_spectra(csv_col_major_with_axis(6))
    assert r.tensor.shape[0] == 6 and r.dropped["wavelength_axis"] == 1


def test_dataset_raw_layout_is_validated_and_keeps_training_parity(csv_dataset_raw):
    r = load_spectra(csv_dataset_raw(6, angle=20))
    assert r.input_format == "dataset_raw" and r.validated is True
    assert r.tensor.shape == (6, 3, 1500)
    assert r.dropped["non_finite"] == 1                    # the label/wavelength column
    # Training parity: the first 5 "bands" are the metadata values (angle, 5, 0, 5, 0) before SNV,
    # so the SNV curve must NOT be smooth across that prefix.
    s0 = r.tensor[0, 0].numpy()
    assert s0[0] > s0[2] > -10 and not np.allclose(s0[:5], s0[5:10])


def test_wrong_band_count_error():
    df = pd.DataFrame(np.random.rand(4, 1000)); buf = io.BytesIO(); df.to_csv(buf, index=False, header=False)
    with pytest.raises(SpectraFormatError, match="spectral bands"):
        load_spectra(buf.getvalue())


def test_flat_line_scans_refused(make_spectra):
    x = make_spectra(5); x[2] = 7.0
    buf = io.BytesIO(); pd.DataFrame(x).to_csv(buf, index=False, header=False)
    with pytest.raises(SpectraFormatError, match="flat-line"):
        load_spectra(buf.getvalue())


def test_all_flat_rejected():
    buf = io.BytesIO(); pd.DataFrame(np.full((3, 1500), 5.0)).to_csv(buf, index=False, header=False)
    with pytest.raises(SpectraFormatError, match="flat-line"):
        load_spectra(buf.getvalue())


def test_nan_scan_refused(make_spectra):
    x = make_spectra(6); x[1, 10] = np.nan; x[3, 20] = np.nan
    buf = io.BytesIO(); pd.DataFrame(x).to_csv(buf, index=False, header=False)
    with pytest.raises(SpectraFormatError, match="finite"):
        load_spectra(buf.getvalue())


def test_too_many_scans_is_rejected_not_subsampled(make_spectra):
    buf = io.BytesIO(); pd.DataFrame(make_spectra(12)).to_csv(buf, index=False, header=False)
    with pytest.raises(SpectraFormatError, match="limit"):
        load_spectra(buf.getvalue(), max_scans=10)


def test_every_scan_reaches_the_model(make_spectra):
    """F1 regression: no random subsampling - 100 scans in, 100 scans out."""
    buf = io.BytesIO(); pd.DataFrame(make_spectra(100)).to_csv(buf, index=False, header=False)
    assert load_spectra(buf.getvalue()).tensor.shape[0] == 100
