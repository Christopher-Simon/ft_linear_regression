"""
This module contains the function to plot data from a data file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_file_data(
    data: pd.DataFrame,
    title: str = "Car Price on Km Driven",
    x_label: str = "Km Driven",
    y_label: str = "Price (€)",
    line_params: tuple = None,
) -> None:
    """
    Plot the data from the data file
    :param data_file: The data file
    :param title: The title of the plot
    :param x_label: The x-axis label
    :param y_label: The y-axis label
    """

    plt.figure(figsize=(8, 5))
    plt.scatter(data["km"], data["price"], color="blue", alpha=0.6, edgecolors="black")

    if line_params:
        a, b = line_params
        x_range = np.linspace(
            data["km"].min(), data["km"].max(), 100
        )
        y_values = a * x_range + b
        plt.plot(
            x_range,
            y_values, 
            color="red",
            linewidth=2,
            linestyle="-",
            label=f"y = {a:.2f}x + {b:.2f}",
        )
        plt.legend()


    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
