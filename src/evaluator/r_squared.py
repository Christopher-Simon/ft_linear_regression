"""
A r-squared evaluator for evaluating the performance of regression models.
"""

from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from visualisation.data_graph import plot_file_data


def r_squared(
    x_list: List[float],
    y_true: List[float],
    line_params: tuple[float, float],
) -> float:
    """
    The r-squared value is a statistical measure that represents the
    proportion of the variance for a dependent variable that's
    explained by an independent variable or variables in a regression
    model.
    SSR is the sum of squares of the residuals, and SST is the total
    sum of squares.
    :param x_list: List of x values
    :param y_true: List of true y values
    :param line_params: Tuple of (slope, intercept) for the line
    :return: R-squared value
    """

    if len(x_list) != len(y_true):
        raise ValueError("x_list and y_true must have the same length.")

    y_mean = sum(y_true) / len(y_true)

    slope, intercept = line_params

    y_predicted = [slope * x + intercept for x in x_list]

    y_mean = sum(y_true) / len(y_true)
    ssr = sum((y_true[i] - y_predicted[i]) ** 2 for i in range(len(y_true)))
    sst = sum((y - y_mean) ** 2 for y in y_true)

    if sst == 0:
        r_squared_value = 0.0
    else:
        r_squared_value = 1 - (ssr / sst)

    plot_file_data(
        data=pd.DataFrame({"km": x_list, "price": y_true}),
        title="R-squared Evaluation",
        x_label="Km Driven",
        y_label="Price (€)",
        line_params=[line_params, (0, y_mean)],
    )
    plt.show()

    return r_squared_value
