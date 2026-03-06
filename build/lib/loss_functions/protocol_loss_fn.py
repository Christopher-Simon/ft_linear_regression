"""
This module defines a protocol for loss functions used in machine learning.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol


class LossFunction(Protocol):
    """
    Protocol for loss functions.
    """

    @abstractmethod
    def loss(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the loss function
        """

    @abstractmethod
    def derived_b(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derivation with respect to b
        """

    @abstractmethod
    def derived_w(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        The formula for the derivation with respect to w
        """
