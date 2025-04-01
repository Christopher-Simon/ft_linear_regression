"""
Evaluator Comparison
This module compares custom evaluation metrics with those from scikit-learn.
"""

from sklearn.metrics import mean_squared_error as mse_sklearn
from sklearn.metrics import mean_absolute_error as mae_sklearn
from sklearn.metrics import r2_score as r2_sklearn


import pandas as pd
from pandas import DataFrame
from model.simple_linear_regression import SimpleLinearRegression
from normalizers.z_score import ZScoreNormalizer
from evaluator.r_squared import r_squared
from evaluator.mean_absolute_error import mean_absolute_error
from loss_functions.mean_squared_error import MeanSquaredError


def evaluator_comparison():
    """
    Compare custom metrics with scikit-learn metrics.
    """
    model = SimpleLinearRegression()
    data: DataFrame = pd.read_csv("data/data.csv")

    z_score_normalizer = ZScoreNormalizer(data)
    z_score_normalizer.get_main_values()
    normalize_data: DataFrame = z_score_normalizer.transform()
    model.fit(
        x_list=normalize_data["km"].tolist(),
        y_list=normalize_data["price"].tolist(),
        normalizer=z_score_normalizer,
    )

    custom_mae = mean_absolute_error(
        model.predict(data["km"].tolist()), data["price"].tolist()
    )

    mse = MeanSquaredError()
    custom_mse = mse.loss(
        x_list=data["km"].tolist(),
        y_list=data["price"].tolist(),
        estimate_func=model.estimate_price,
    )

    custom_r2 = r_squared(
        data["km"].tolist(),
        data["price"].tolist(),
        (model.slope, model.intercept),
    )

    y_pred = model.predict(data["km"].tolist())

    sklearn_mse = mse_sklearn(data["price"].tolist(), y_pred)
    sklearn_mae = mae_sklearn(data["price"].tolist(), y_pred)
    sklearn_r2 = r2_sklearn(data["price"].tolist(), y_pred)

    print(f"Custom Mean Absolute Error: {custom_mae:.2f} €")
    print(f"Scikit-Learn MAE:          {sklearn_mae:.2f} €\n")

    print(f"Custom Mean Squared Error: {custom_mse:.2f}")
    print(f"Scikit-Learn MSE:          {sklearn_mse:.2f}\n")

    print(f"Custom R-squared:          {custom_r2:.2%}")
    print(f"Scikit-Learn R^2:          {sklearn_r2:.2%}")
