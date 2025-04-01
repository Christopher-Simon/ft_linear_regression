"""
Module for Mean Squared Error (MSE) loss function.
This module provides a class to calculate the Mean Squared Error
between observed and predicted values.
It includes methods to compute the MSE, its derivative with respect to
the slope and intercept, and the sum of squared residuals.
"""

from typing import Callable

from loss_functions.protocol_loss_fn import LossFunction


class MeanSquaredError(LossFunction):
    """
    Class to calculate the Mean Squared Error (MSE) loss function.
    """

    def loss(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        Calculate the Mean Squared Error (MSE) between observed and predicted\
        values.
        :param x_list: List of input values
        :param y_list: List of observed values
        :param estimate_func: Function to estimate the predicted values
        :return: Mean Squared Error
        """
        sum_squared_residuals = 0
        m = len(x_list)
        for x, y in zip(x_list, y_list):
            residual = estimate_func(x) - y
            sum_squared_residuals += residual**2
        return sum_squared_residuals / m

    def derived_b(
        self,
        x_list: list[float],
        y_list: list[float],
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

    def derived_w(
        self,
        x_list: list[float],
        y_list: list[float],
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
