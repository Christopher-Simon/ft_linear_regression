"""
Module for Mean Squared Error (MSE) loss function.
This module provides a class to calculate the Mean Squared Error
between observed and predicted values.
It includes methods to compute the MSE, its derivative with respect to
the slope and intercept, and the sum of squared residuals.
"""

from typing import Callable


class MeanSquaredError:
    """
    Class to calculate the Mean Squared Error (MSE) loss function.
    """

    def derived_mse_b(
        self,
        x_list: list,
        y_list: list,
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derived MSE with respect to b is:
        dMSE/db = 1/M * sum((a * x + b) - y)
        where M is the number of data points.
        This function calculates the derived MSE with respect to b.
        """

        sum_d_f_b = 0
        m = len(x_list)
        for x, y in zip(x_list, y_list):
            d_f_b = estimate_func(x) - y
            sum_d_f_b += d_f_b
        return sum_d_f_b / m

    def derived_mse_a(
        self,
        x_list: list,
        y_list: list,
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derived MSE with respect to a is:
        dMSE/da = 1/M * sum((a * x + b) - y) * x
        where M is the number of data points.
        This function calculates the derived MSE with respect to a.
        """

        sum_d_f_w = 0
        m = len(x_list)
        for x, y in zip(x_list, y_list):
            d_f_w = (estimate_func(x) - y) * x
            sum_d_f_w += d_f_w
        return sum_d_f_w / m
