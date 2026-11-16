import numpy as np

from numcompute_stream.preprocessing import StandardScaler


def test_partial_fit_scales_tiny_values_and_ignores_nan():
    scaler = StandardScaler().partial_fit([[1e-200, np.nan]])
    scaler.partial_fit([[3e-200, 7.0]])

    result = scaler.transform([[1e-200, np.nan], [3e-200, 7.0]])

    np.testing.assert_allclose(result, [[-1.0, np.nan], [1.0, 0.0]], equal_nan=True)