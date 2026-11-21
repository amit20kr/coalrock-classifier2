import pytest
import numpy as np
import pandas as pd
import io
from backend.pipeline import process_csv_to_tensor

def test_pipeline_parity():
    # Simulate a raw CSV block
    np.random.seed(42)
    raw_data = np.random.rand(10, 1500)
    df = pd.DataFrame(raw_data)
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False, header=False)
    
    tensor, wavelengths = process_csv_to_tensor(csv_buf.getvalue())
    
    # Check shape
    assert tensor.shape == (10, 3, 1500)
    
    # Check SNV channel (channel 0) zero mean, unit variance
    snv = tensor[:, 0, :].numpy()
    means = snv.mean(axis=1)
    stds = snv.std(axis=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(stds, 1.0, atol=1e-5)
    
    # Wavelengths should be 1500 length
    assert len(wavelengths) == 1500
    assert wavelengths[0] == 1000.0
    assert wavelengths[-1] == 2499.0


    # Independent reproduction of the training arithmetic for all three channels.
    from scipy.signal import savgol_filter
    values = raw_data.astype(np.float32)
    reference = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-8)
    expected = np.stack([reference, savgol_filter(reference, 11, 2, deriv=1, axis=1),
                         savgol_filter(reference, 11, 2, deriv=2, axis=1)], axis=1)
    np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-6, atol=1e-6)
