"""
module for the simple linear regression model
"""

from loss_functions.mean_squared_error import MeanSquaredError
from loss_functions.protocol import LossFunction


class SimpleLinearRegression:
    """
    Simple linear regression model
    """

    elements: list[float] = ["a", "b"]

    def __init__(
        self,
        slope: float = 0,
        intercept: float = 0,
        loss_fn: LossFunction = MeanSquaredError(),
    ) -> None:
        self.slope = slope
        self.intercept = intercept
        self.loss_fn = loss_fn

    def estimate_price(self, x: float) -> float:
        """
        Estimate the price of a car given the km driven
        :param x: The target value
        :return: The estimated price
        """
        return self.intercept + self.slope * x

    