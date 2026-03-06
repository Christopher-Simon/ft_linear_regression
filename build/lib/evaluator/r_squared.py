"""
A r-squared evaluator for evaluating the performance of regression models.
"""


def r_squared(
    x_list: list[float],
    y_true: list[float],
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

    if len(y_true) == 0:
        raise ValueError("Input lists cannot be empty.")

    y_mean = sum(y_true) / len(y_true)

    slope, intercept = line_params

    y_predicted = [slope * x + intercept for x in x_list]

    ssr = sum((y_true[i] - y_predicted[i]) ** 2 for i in range(len(y_true)))
    sst = sum((y - y_mean) ** 2 for y in y_true)

    if sst == 0:
        r_squared_value = 0.0
    else:
        r_squared_value = 1 - (ssr / sst)

    return r_squared_value
