"""
Main module for the application.
"""

import pandas as pd
from pandas import DataFrame

from model.simple_linear_regression import SimpleLinearRegression
from normalizers.z_score import ZScoreNormalizer
from normalizers.minmax_normalizer import MinMaxNormalizer
from evaluator.r_squared import r_squared


def main():
    """
    Main function to run the application.
    """

    data: DataFrame = pd.read_csv("data/data.csv")

    z_score_normalizer = ZScoreNormalizer(data)

    normalized_data: DataFrame = z_score_normalizer.transform()

    custom_model_z = SimpleLinearRegression()
    custom_model_z.fit(
        x_list=normalized_data["km"].tolist(),
        y_list=normalized_data["price"].tolist(),
        normalizer=z_score_normalizer,
    )

    minmax_normalizer = MinMaxNormalizer(data)

    normalized_data: DataFrame = minmax_normalizer.transform()

    custom_model_m = SimpleLinearRegression()
    custom_model_m.fit(
        x_list=normalized_data["km"].tolist(),
        y_list=normalized_data["price"].tolist(),
        normalizer=minmax_normalizer,
        learning_rate=0.1,
    )

    res = r_squared(
        x_list=data["km"].tolist(),
        y_true=data["price"].tolist(),
        line_params=(custom_model_z.slope, custom_model_z.intercept),
    )
    print("r-squared: ", res)


if __name__ == "__main__":
    main()
