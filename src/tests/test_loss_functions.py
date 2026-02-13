"""
Unit tests for the loss functions (MSE, MAE, SSE).
"""

from collections.abc import Callable

import pytest

from loss_functions.mean_absolute_error import MeanAbsoluteError
from loss_functions.mean_squared_error import MeanSquaredError
from loss_functions.sum_squared_error import SumSquaredError


@pytest.fixture
def sample_data() -> tuple[list[float], list[float]]:
    """
    Fixture for simple X and Y data.
    X = [1, 2, 3]
    Y = [2, 3, 5]
    """
    return [1.0, 2.0, 3.0], [2.0, 3.0, 5.0]


@pytest.fixture
def estimate_func() -> Callable[[float], float]:
    """
    Fixture for a simple estimate function y = 2x.
    Predictions will be: [2, 4, 6]
    """
    return lambda x: 2.0 * x


def test_mean_squared_error(
    sample_data: tuple[list[float], list[float]],
    estimate_func: Callable[[float], float],
) -> None:
    """
    Test Mean Squared Error (MSE).
    Preds: [2, 4, 6]
    Reals: [2, 3, 5]
    Diffs: [0, 1, 1]
    Squared: [0, 1, 1] -> Sum: 2 -> Mean: 2/3
    """
    x, y = sample_data
    mse = MeanSquaredError()

    # 1. Test Loss
    expected_loss = 2.0 / 3.0
    assert mse.loss(x, y, estimate_func) == pytest.approx(expected_loss)

    # 2. Test Derivative w.r.t bias (b)
    # dJ/db = 1/m * sum(pred - y)
    # Diffs: 0, 1, 1 -> Sum: 2 -> Mean: 2/3
    expected_db = 2.0 / 3.0
    assert mse.derived_b(x, y, estimate_func) == pytest.approx(expected_db)

    # 3. Test Derivative w.r.t weight (w)
    # dJ/dw = 1/m * sum((pred - y) * x)
    # Diffs * x: 0*1, 1*2, 1*3 -> [0, 2, 3] -> Sum: 5 -> Mean: 5/3
    expected_dw = 5.0 / 3.0
    assert mse.derived_w(x, y, estimate_func) == pytest.approx(expected_dw)


def test_mean_absolute_error(
    sample_data: tuple[list[float], list[float]],
    estimate_func: Callable[[float], float],
) -> None:
    """
    Test Mean Absolute Error (MAE).
    Preds: [2, 4, 6]
    Reals: [2, 3, 5]
    Diffs: [0, 1, 1]
    Abs:   [0, 1, 1] -> Sum: 2 -> Mean: 2/3
    """
    x, y = sample_data
    mae = MeanAbsoluteError()

    # 1. Test Loss
    expected_loss = 2.0 / 3.0
    assert mae.loss(x, y, estimate_func) == pytest.approx(expected_loss)

    # 2. Test Derivative w.r.t bias (b)
    # dJ/db = 1/m * sum(sgn(pred - y))
    # Signs: 0, 1, 1 -> Sum: 2 -> Mean: 2/3
    expected_db = 2.0 / 3.0
    assert mae.derived_b(x, y, estimate_func) == pytest.approx(expected_db)

    # 3. Test Derivative w.r.t weight (w)
    # dJ/dw = 1/m * sum(sgn(pred - y) * x)
    # Signs * x: 0*1, 1*2, 1*3 -> [0, 2, 3] -> Sum: 5 -> Mean: 5/3
    expected_dw = 5.0 / 3.0
    assert mae.derived_w(x, y, estimate_func) == pytest.approx(expected_dw)


def test_sum_squared_error(
    sample_data: tuple[list[float], list[float]],
    estimate_func: Callable[[float], float],
) -> None:
    """
    Test Sum Squared Error (SSE).
    Preds: [2, 4, 6]
    Reals: [2, 3, 5]
    Diffs: [0, 1, 1]
    Squared: [0, 1, 1] -> Sum: 2
    """
    x, y = sample_data
    sse = SumSquaredError()

    # 1. Test Loss
    expected_loss = 2.0
    assert sse.loss(x, y, estimate_func) == pytest.approx(expected_loss)

    # 2. Test Derivative w.r.t bias (b)
    # dJ/db = sum(2 * (pred - y))
    # 2 * Diffs: 0, 2, 2 -> Sum: 4
    expected_db = 4.0
    assert sse.derived_b(x, y, estimate_func) == pytest.approx(expected_db)

    # 3. Test Derivative w.r.t weight (w)
    # dJ/dw = sum(2 * (pred - y) * x)
    # 2 * Diffs * x: 0, 2*2, 2*3 -> [0, 4, 6] -> Sum: 10
    expected_dw = 10.0
    assert sse.derived_w(x, y, estimate_func) == pytest.approx(expected_dw)
