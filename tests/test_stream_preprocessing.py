import numpy as np

from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, Imputer


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

def test_imputer_partial_fit_weights_each_observation():
    imputer = Imputer(fill_value=-1).partial_fit([[2.0, np.nan]])
    np.testing.assert_allclose(imputer.statistics_, [2.0, -1.0])

    imputer.partial_fit([[4.0, 8.0], [6.0, np.nan], [8.0, 12.0]])
    np.testing.assert_allclose(
        imputer.transform([[np.nan, np.nan]]),
        [[5.0, 10.0]],
    )