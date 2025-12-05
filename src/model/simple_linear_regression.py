"""
module for the simple linear regression model
"""

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from loss_functions.mean_squared_error import MeanSquaredError
from loss_functions.protocol_loss_fn import LossFunction
from normalizers.protocol_normalizers import Normalizer


def render_plot(
    ax: Axes,
    data: pd.DataFrame,
    line_params: tuple[float, float] | None = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
):
    """
    Render the plot
    """
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(data["km"], data["price"], color="blue", alpha=0.6, edgecolors="black")

    if line_params:
        a, b = line_params
        x_range = np.linspace(data["km"].min(), data["km"].max(), 100)
        y_values = a * x_range + b
        ax.plot(
            x_range,
            y_values,
            color="red",
            linewidth=2,
            linestyle="-",
            label=f"y = {a:.2f}x + {b:.2f}",
        )
        ax.legend()


class SimpleLinearRegression:
    """
    Simple linear regression model
    """

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

    def fit(
        self,
        x_list: list[float],
        y_list: list[float],
        normalizer: Normalizer,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        min_loss: float = 1e-5,
    ) -> None:
        """
        Fit the model to the data
        :param x_list: The km driven
        :param y_list: The price of the car
        :param learning_rate: The learning rate
        :param iterations: The number of iterations
        """

        tmp_loss = None
        loss_values = []

        plt.ion()
        _, ax_fit = plt.subplots(figsize=(8, 5))
        fig_loss, ax_loss = plt.subplots(figsize=(8, 5))

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

            t_b = learning_rate * d_b
            t_w = learning_rate * d_w

            self.intercept -= t_b
            self.slope -= t_w

            if i % 10 == 0:
                loss_values.append(loss)
                ax_loss.clear()
                ax_loss.set_title("Loss over iterations")
                ax_loss.set_xlabel("Iteration")
                ax_loss.set_ylabel("Loss")
                ax_loss.scatter(
                    range(len(loss_values)), loss_values, label="Loss", marker="o", s=40
                )
                ax_loss.legend()

                fig_loss.canvas.draw()
                fig_loss.canvas.flush_events()
                plt.pause(0.0001)

            render_plot(
                ax=ax_fit,
                data=normalizer.normalized_data,
                line_params=(self.slope, self.intercept),
                title="Car Price on Km Driven",
                x_label="Km Driven",
                y_label="Price (€)",
            )
            plt.pause(0.0001)

        plt.ioff()
        plt.show()

        original_slope, original_intercept = normalizer.invert_weight_bias(
            self.slope, self.intercept
        )

        self.slope = original_slope
        self.intercept = original_intercept

        plt.show()

    def predict(self, x_list: list[float]) -> list[float]:
        """
        Predict the price of a car given the km driven
        :param x_list: The km driven
        :return: The predicted price
        """
        return [self.estimate_price(x) for x in x_list]
