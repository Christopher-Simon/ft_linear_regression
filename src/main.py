"""
Main module for the application.
"""

import pandas as pd
from pandas import DataFrame

from model.simple_linear_regression import SimpleLinearRegression
from normalizers.z_score import ZScoreNormalizer


def main():
    """
    Main function to run the application.
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


if __name__ == "__main__":
    main()
