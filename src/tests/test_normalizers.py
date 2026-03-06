"""
Unit tests for the refactored Normalizers (Z-Score and Min-Max).
Includes comparison against Scikit-Learn's preprocessing scalers.
"""

import numpy as np
import numpy.typing as npt

import pytest
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from src.normalizers.minmax_normalizer import MinMaxNormalizer
from src.normalizers.z_score import ZScoreNormalizer


@pytest.fixture
def sample_data() -> npt.NDArray[np.float64]:
    """
    Simple dataset with enough variance for testing.
    """
    return np.array([0.0, 5.0, 10.0, 15.0, 20.0])


def test_zscore_vs_sklearn(sample_data: npt.NDArray[np.float64]) -> None:
    """
    Compare custom ZScoreNormalizer against Scikit-Learn's StandardScaler.
    """
    # 1. Custom Normalizer
    custom_norm = ZScoreNormalizer()
    custom_transformed = custom_norm.fit_transform(sample_data)

    # 2. Scikit-Learn Normalizer
    sk_norm = SkStandardScaler()
    # sklearn expects 2D arrays (n_samples, n_features), so we reshape it
    sk_transformed = sk_norm.fit_transform(sample_data.reshape(-1, 1)).flatten()

    # 3. Compare the transformed arrays
    np.testing.assert_array_almost_equal(custom_transformed, sk_transformed)

    # 4. Compare the Inverse Transform
    custom_inverted = custom_norm.inverse_transform(custom_transformed)
    sk_inverted = sk_norm.inverse_transform(sk_transformed.reshape(-1, 1)).flatten()

    np.testing.assert_array_almost_equal(custom_inverted, sk_inverted)


def test_minmax_vs_sklearn(sample_data: npt.NDArray[np.float64]) -> None:
    """
    Compare custom MinMaxNormalizer against Scikit-Learn's MinMaxScaler.
    """
    # 1. Custom Normalizer
    custom_norm = MinMaxNormalizer()
    custom_transformed = custom_norm.fit_transform(sample_data)

    # 2. Scikit-Learn Normalizer
    sk_norm = SkMinMaxScaler()
    sk_transformed = sk_norm.fit_transform(sample_data.reshape(-1, 1)).flatten()

    # 3. Compare transformed arrays
    np.testing.assert_array_almost_equal(custom_transformed, sk_transformed)

    # 4. Compare the Inverse Transform
    custom_inverted = custom_norm.inverse_transform(custom_transformed)
    sk_inverted = sk_norm.inverse_transform(sk_transformed.reshape(-1, 1)).flatten()

    np.testing.assert_array_almost_equal(custom_inverted, sk_inverted)


# --- EDGE CASES ---


def test_empty_array() -> None:
    """
    Test that fitting an empty array raises ValueError.
    """
    empty_data = np.array([])

    # 1. MinMax
    # np.min([]) raises ValueError automatically, so this should pass as-is.
    norm_mm = MinMaxNormalizer()
    with pytest.raises(ValueError):
        norm_mm.fit(empty_data)

    # 2. ZScore
    # np.mean([]) returns NaN and warns, but typically we want a ValueError
    # to stop invalid training early.
    norm_zs = ZScoreNormalizer()
    with pytest.raises(ValueError):
        norm_zs.fit(empty_data)


def test_constant_array() -> None:
    """
    Test constant data (e.g. [5, 5, 5]).
    Standard deviation is 0, Range is 0.
    Should handle division by zero gracefully.
    """
    data = np.array([5.0, 5.0, 5.0])

    # 1. MinMax
    norm_mm = MinMaxNormalizer()
    norm_mm.fit(data)
    res_mm = norm_mm.transform(data)

    # Range is 0, handled to 1.0. (5-5)/1 = 0.
    assert np.allclose(res_mm, 0.0)
    # Inversion should return original 5s
    assert np.allclose(norm_mm.inverse_transform(res_mm), data)

    # 2. ZScore
    norm_zs = ZScoreNormalizer()
    norm_zs.fit(data)
    res_zs = norm_zs.transform(data)

    # Std is 0, handled to 1.0. (5-5)/1 = 0.
    assert np.allclose(res_zs, 0.0)
    assert np.allclose(norm_zs.inverse_transform(res_zs), data)


def test_single_value() -> None:
    """
    Test a single data point. Effectively a constant array.
    """
    data = np.array([42.0])

    # 1. MinMax
    norm_mm = MinMaxNormalizer()
    norm_mm.fit(data)
    res_mm = norm_mm.transform(data)

    assert np.allclose(res_mm, 0.0)
    assert np.allclose(norm_mm.inverse_transform(res_mm), data)

    # 2. ZScore
    norm_zs = ZScoreNormalizer()
    norm_zs.fit(data)
    res_zs = norm_zs.transform(data)

    assert np.allclose(res_zs, 0.0)
    assert np.allclose(norm_zs.inverse_transform(res_zs), data)


def test_nan_input() -> None:
    """
    Test that NaNs propagate gracefully (or ensure they don't crash).
    With current logic, they propagate as NaNs, which is standard NumPy behavior.
    """
    data = np.array([1.0, 2.0, np.nan])

    # 1. MinMax
    norm_mm = MinMaxNormalizer()
    norm_mm.fit(data)
    # np.min/max with NaN results in NaN, so transform returns NaNs.
    res_mm = norm_mm.transform(data)
    assert np.isnan(res_mm).all()

    # 2. ZScore
    norm_zs = ZScoreNormalizer()
    norm_zs.fit(data)
    # np.mean/std with NaN results in NaN.
    res_zs = norm_zs.transform(data)
    assert np.isnan(res_zs).all()
