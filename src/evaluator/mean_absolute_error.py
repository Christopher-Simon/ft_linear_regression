"""
Module to calculate the Mean Absolute Error (MAE) between two lists of numbers.
"""


def mean_absolute_error(
    y_true: list[float],
    y_predicted: list[float],
):
    """
    Calculate the Mean Absolute Error (MAE) between two lists of numbers.
    :param y_true: List of input values
    :param y_predicted: List of observed values
    :return: Mean Absolute Error
    """
    if len(y_true) != len(y_predicted):
        raise ValueError("y_true and y_predicted must have the same length.")
    return sum(abs(y - y_predicted[i]) for i, y in enumerate(y_true)) / len(y_true)
