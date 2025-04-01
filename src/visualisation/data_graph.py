"""
This module contains the function to plot data from a data file.
"""

from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_file_data(
    data: pd.DataFrame,
    title: str = "Car Price on Km Driven",
    x_label: str = "Km Driven",
    y_label: str = "Price (€)",
    line_params: List[tuple[float, float]] | None = None,
) -> None:
    """
    Plot the data from the data file
    :param data_file: The data file
    :param title: The title of the plot
    :param x_label: The x-axis label
    :param y_label: The y-axis label
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(
        x=data["km"], y=data["price"], color="blue", alpha=0.6, edgecolors="black"
    )

    if line_params:
        for i, (a, b) in enumerate(line_params):
            x_range = np.linspace(data["km"].min(), data["km"].max(), 100)
            y_values = a * x_range + b
            plt.plot(
                x_range,
                y_values,
                color=COLORS[i % len(COLORS)],
                linewidth=2,
                linestyle="-",
                label=f"y = {a:.2f}x + {b:.2f}",
            )
        plt.legend()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
