import numpy as np

from numcompute_stream.stats import StreamStats


def test_moments_and_quantiles_across_uneven_chunks():
    stats = StreamStats([0, 3, 6])
    stats.update_stats([[1.0, np.nan]])
    stats.update_stats([[2.0, 4.0], [5.0, 6.0]])

    np.testing.assert_allclose(stats.mean, [8 / 3, 5.0])
    np.testing.assert_allclose(stats.var(), [26 / 9, 1.0])
    np.testing.assert_allclose(stats.quantile(0.5), [2.0, 5.0])


def test_histogram_adds_counts_per_column():
    stats = StreamStats([0, 3, 6])
    stats.update_stats([[1.0, np.nan], [4.0, 2.0]])
    stats.update_stats([[3.0, 6.0]])

    np.testing.assert_array_equal(stats.histogram(), [[1, 2], [1, 1]])