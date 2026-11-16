import numpy as np

from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler


def test_partial_fit_scales_tiny_values_and_ignores_nan():
    scaler = StandardScaler().partial_fit([[1e-200, np.nan]])
    scaler.partial_fit([[3e-200, 7.0]])

    result = scaler.transform([[1e-200, np.nan], [3e-200, 7.0]])

    np.testing.assert_allclose(result, [[-1.0, np.nan], [1.0, 0.0]], equal_nan=True)

def test_minmax_partial_fit_updates_ranges_across_chunks():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.partial_fit([[2.0, np.nan], [4.0, np.nan]])
    scaler.partial_fit([[0.0, 10.0], [8.0, 20.0]])

    np.testing.assert_allclose(
        scaler.transform([[0.0, 10.0], [4.0, 15.0], [8.0, 20.0]]),
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
    )