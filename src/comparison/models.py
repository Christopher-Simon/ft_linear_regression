"""
comparing linear regression models, mean squared error
"""

# Scikit-learn imports:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae_sklearn
from sklearn.metrics import r2_score as r2_sklearn

import pandas as pd
from pandas import DataFrame

from evaluator.mean_absolute_error import mean_absolute_error
from evaluator.r_squared import r_squared
from model.simple_linear_regression import SimpleLinearRegression
from normalizers.z_score import ZScoreNormalizer


def model_comparison():
    """
    Compare custom metrics with scikit-learn metrics.
    """
    data: DataFrame = pd.read_csv("data/data.csv")

    z_score_normalizer = ZScoreNormalizer(data)
    z_score_normalizer.get_main_values()

    normalized_data: DataFrame = z_score_normalizer.transform()

    custom_model = SimpleLinearRegression()
    custom_model.fit(
        x_list=normalized_data["km"].tolist(),
        y_list=normalized_data["price"].tolist(),
        normalizer=z_score_normalizer,
    )

    custom_predictions = custom_model.predict(data["km"].tolist())

    custom_mse = mean_squared_error(data["price"].tolist(), custom_predictions)
    print(f"Custom model MSE: {custom_mse:.2f}")
    custom_mae = mean_absolute_error(data["price"].tolist(), custom_predictions)
    print(f"Custom model MAE: {custom_mae:.2f} €")
    custom_r2 = r_squared(
        data["km"].tolist(),
        data["price"].tolist(),
        (custom_model.slope, custom_model.intercept),
    )
    print(f"Custom model R²: {custom_r2:.2f}")

    x_list = data[["km"]]
    y_list = data["price"]
    sk_model = LinearRegression()
    sk_model.fit(x_list, y_list)

    sk_predictions = sk_model.predict(x_list)

    sk_mse = mean_squared_error(y_list, sk_predictions)
    print(f"Scikit-learn MSE: {sk_mse:.2f}")
    sk_mae = mae_sklearn(y_list, sk_predictions)
    print(f"Scikit-learn MAE: {sk_mae:.2f} €")
    sk_r2 = r2_sklearn(y_list, sk_predictions)
    print(f"Scikit-learn R²: {sk_r2:.2f}")
