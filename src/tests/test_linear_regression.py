"""
Unit tests and edge cases for the SimpleLinearRegression model.
Includes comparison with Scikit-Learn.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SkLinearRegression

from loss_functions.mean_squared_error import MeanSquaredError
from model.simple_linear_regression import SimpleLinearRegression


@pytest.fixture
def model() -> SimpleLinearRegression:
    """Returns a fresh model instance for each test."""
    return SimpleLinearRegression(loss_fn=MeanSquaredError())


def test_compare_with_sklearn(model: SimpleLinearRegression) -> None:
    """
    Compare custom implementation against Scikit-Learn's LinearRegression.
    """
    # 1. Prepare Data
    # Simple dataset (small numbers to avoid normalization issues with basic GD)
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 5.0, 4.0, 5.0])

    # 2. Train Scikit-Learn Model
    # Sklearn expects shape (n_samples, n_features) for x
    sk_model = SkLinearRegression()
    sk_model.fit(x, y)

    # 3. Train Custom Model
    # Convert inputs to list[float] as per your model's requirement
    # We use a higher iteration count to ensure Gradient Descent gets close to the exact OLS solution
    model.fit(
        x_list=x.flatten().tolist(),
        y_list=y.tolist(),
        learning_rate=0.05,
        iterations=10000,
    )

    # 4. Compare Results
    # Check Slope (Coefficient)
    # Allow a small relative tolerance (1%) because GD is approximate vs OLS exact solution
    assert model.slope == pytest.approx(float(sk_model.coef_[0]), rel=5e-2)

    # Check Intercept
    assert model.intercept == pytest.approx(float(sk_model.intercept_), rel=5e-2)

    # 5. Compare Predictions
    x_test = np.array([3.5])

    # Custom prediction (returns list)
    custom_pred = model.predict(x_test.tolist())[0]

    # Sklearn prediction (returns array)
    sk_pred = sk_model.predict(x_test.reshape(-1, 1))[0]

    assert custom_pred == pytest.approx(sk_pred, rel=5e-2)


def test_fit_perfect_line(model: SimpleLinearRegression) -> None:
    """
    Test fitting a perfect line y = 2x + 1.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    # Fit model (convert numpy arrays to lists)
    model.fit(x.tolist(), y.tolist(), learning_rate=0.1, iterations=1000)

    assert model.slope == pytest.approx(2.0, abs=0.1)
    assert model.intercept == pytest.approx(1.0, abs=0.1)

    preds = model.predict([4.0])
    assert preds[0] == pytest.approx(9.0, abs=0.2)


def test_fit_horizontal_line(model: SimpleLinearRegression) -> None:
    """
    Test fitting a horizontal line y = 5 (Slope should be 0).
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([5.0, 5.0, 5.0, 5.0])

    model.fit(x.tolist(), y.tolist(), learning_rate=0.1, iterations=500)

    assert model.slope == pytest.approx(0.0, abs=0.1)
    assert model.intercept == pytest.approx(5.0, abs=0.1)


def test_predict_shape(model: SimpleLinearRegression) -> None:
    """
    Test that predict returns the correct shape/type.
    """
    model.slope = 2.0
    model.intercept = 1.0

    x_input = np.array([10.0, 20.0])
    preds = model.predict(x_input.tolist())

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0] == 21.0
    assert preds[1] == 41.0


# --- EDGE CASES ---


def test_fit_mismatched_shapes(model: SimpleLinearRegression) -> None:
    """
    Test that providing x and y with different lengths raises an error.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        model.fit(x.tolist(), y.tolist())


def test_fit_empty_data(model: SimpleLinearRegression) -> None:
    """
    Test fitting with empty lists.
    """
    with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
        model.fit([], [])


def test_fit_nan_values(model: SimpleLinearRegression) -> None:
    """
    Test fitting with NaN values.
    """
    x = np.array([1.0, 2.0, np.nan])
    y = np.array([1.0, 2.0, 3.0])

    # Convert to list (np.nan is a valid float)
    model.fit(x.tolist(), y.tolist(), iterations=10)

    assert np.isnan(model.slope) or np.isnan(model.intercept)


def test_fit_single_point(model: SimpleLinearRegression) -> None:
    """
    Test fitting on a single data point.
    """
    x = np.array([2.0])
    y = np.array([4.0])

    model.fit(x.tolist(), y.tolist(), iterations=100)

    pred = model.predict([2.0])
    assert pred[0] == pytest.approx(4.0, abs=0.5)
