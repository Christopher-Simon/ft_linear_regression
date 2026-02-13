"""
Module for Sum Squared Error (SSE) loss function.
This module provides a class to calculate the Sum Squared Error
between observed and predicted values.
It includes methods to compute the SSE, its derivative with respect to
the slope and intercept, and the sum of squared residuals.
"""

from collections.abc import Callable

from loss_functions.protocol_loss_fn import LossFunction


class SumSquaredError(LossFunction):
    """
    Class to calculate the Sum Squared Error (SSE) loss function.
    Formula: J = sum((pred - y)^2)
    """

    def loss(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        Calculate the Sum Squared Error (SSE) between observed and predicted
        values.
        """
        sum_squared_residuals: float = 0
        for x, y in zip(x_list, y_list):
            residual = estimate_func(x) - y
            sum_squared_residuals += residual**2
        return sum_squared_residuals

    def derived_b(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derived SSE with respect to b is:
        dSSE/db = sum(2 * (y_pred - y))
        """
        sum_d_f_b: float = 0
        for x, y in zip(x_list, y_list):
            d_f_b = 2 * (estimate_func(x) - y)
            sum_d_f_b += d_f_b
        return sum_d_f_b

    def derived_w(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derived SSE with respect to w is:
        dSSE/dw = sum(2 * (y_pred - y) * x)
        """
        sum_d_f_w: float = 0
        for x, y in zip(x_list, y_list):
            d_f_w = 2 * (estimate_func(x) - y) * x
            sum_d_f_w += d_f_w
        return sum_d_f_w
