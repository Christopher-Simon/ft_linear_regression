"""
This module defines a protocol for loss functions used in machine learning.
"""

from typing import Callable, Protocol


class LossFunction(Protocol):
    """
    Protocol for loss functions.
    """

    def calculate(self, y_true: list[float], y_pred: list[float]) -> float:
        """Calculate the loss between true and predicted values"""
        ...

    def derivative(self, y_true: list[float], y_pred: list[float]) -> list[float]:
        """Calculate the derivative of the loss function"""
        ...

    def derived_b(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derivation with respect to b
        """
        ...

    def derived_w(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derivation with respect to w
        """
        ...
