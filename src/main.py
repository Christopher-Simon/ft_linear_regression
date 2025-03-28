"""
Main module for the application.
"""

import pandas as pd
from pandas import DataFrame
from model.simple_linear_regression import SimpleLinearRegression


def main():
    """Entry point of the application."""
    data: DataFrame = pd.read_csv("data/data_test.csv")
    model = SimpleLinearRegression()


if __name__ == "__main__":
    main()
