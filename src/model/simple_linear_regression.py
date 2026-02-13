"""
module for the simple linear regression model
"""

from collections.abc import Callable
from dataclasses import dataclass

from loss_functions.mean_squared_error import MeanSquaredError
from loss_functions.protocol_loss_fn import LossFunction
from visualisation.data_graph import TrainingStep


@dataclass
class ModelWeights:
    """Dataclass to hold the final trained weights."""

    slope: float
    intercept: float


class SimpleLinearRegression:
    """
    Simple linear regression model
    """

    def __init__(
        self,
        slope: float = 0.0,
        intercept: float = 0.0,
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

    def fit(
        self,
        x_list: list[float],
        y_list: list[float],
        learning_rate: float = 0.01,
        iterations: int = 1000,
        min_loss: float = 1e-5,
        callback: Callable[[TrainingStep], None] | None = None,
    ) -> ModelWeights:
        """
        Fit the model to the data
        :param x_list: The km driven
        :param y_list: The price of the car
        :param learning_rate: The learning rate
        :param iterations: The number of iterations
        :param min_loss: Convergence threshold
        :param callback: Optional callback for real-time visualization
        """

        tmp_loss = None

        for i in range(iterations):
            loss = self.loss_fn.loss(x_list, y_list, self.estimate_price)

            if tmp_loss and abs(loss - tmp_loss) < min_loss:
                print(f"Converged after {i} iterations")
                break

            if tmp_loss and loss < min_loss:
                print(f"Too big learning rate : Converged after {i} iterations")
                break

            tmp_loss = loss
            d_b = self.loss_fn.derived_b(x_list, y_list, self.estimate_price)
            d_w = self.loss_fn.derived_w(x_list, y_list, self.estimate_price)

            if callback and i % 5 == 0:
                step_data = TrainingStep(
                    step=i, w=self.slope, b=self.intercept, grad_w=d_w, grad_b=d_b
                )
                callback(step_data)

            t_b = learning_rate * d_b
            t_w = learning_rate * d_w

            self.intercept -= t_b
            self.slope -= t_w

        if callback:
            final_d_b = self.loss_fn.derived_b(x_list, y_list, self.estimate_price)
            final_d_w = self.loss_fn.derived_w(x_list, y_list, self.estimate_price)
            callback(
                TrainingStep(
                    step=iterations,
                    w=self.slope,
                    b=self.intercept,
                    grad_w=final_d_w,
                    grad_b=final_d_b,
                )
            )

        return ModelWeights(slope=self.slope, intercept=self.intercept)

    def predict(self, x_list: list[float]) -> list[float]:
        """
        Predict the price of a car given the km driven
        :param x_list: The km driven
        :return: The predicted price
        """
        return [self.estimate_price(x) for x in x_list]
