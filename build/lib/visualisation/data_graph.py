from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.normalizers.minmax_normalizer import MinMaxParams
from src.normalizers.protocol_normalizers import Normalizer
from src.normalizers.z_score import ZScoreParams

Normalizers = Normalizer[ZScoreParams] | Normalizer[MinMaxParams]


@dataclass
class VisualizationData:
    """Dataclass to hold all the pre-calculated data needed for plotting."""

    x: np.ndarray
    y: np.ndarray
    x_raw: np.ndarray
    y_raw: np.ndarray
    w: np.ndarray
    b: np.ndarray
    z: np.ndarray
    w_range: np.ndarray
    b_range: np.ndarray
    km_normalizer: Normalizers
    price_normalizer: Normalizers


@dataclass
class VisualizationState:
    """Dataclass to hold the Matplotlib figures and the plot data."""

    fig: Figure
    axs: np.ndarray
    data: VisualizationData


@dataclass
class TrainingStep:
    """Dataclass to hold the current state of the model during training."""

    step: int
    w: float
    b: float
    grad_w: float
    grad_b: float


def init_visualization(
    x: np.ndarray,
    y: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    km_normalizer: Normalizers,
    price_normalizer: Normalizers,
) -> VisualizationState:
    """
    Initializes the matplotlib figure and pre-calculates the cost surface.
    Returns a populated VisualizationState dataclass.
    """
    w_range = np.linspace(-2, 2, 50)
    b_range = np.linspace(-2, 2, 50)
    w, b = np.meshgrid(w_range, b_range)
    z = np.zeros_like(w)

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            z[i, j] = np.sum(((w[i, j] * x + b[i, j]) - y) ** 2)

    plot_data = VisualizationData(
        x=x,
        y=y,
        x_raw=x_raw,
        y_raw=y_raw,
        w=w,
        b=b,
        z=z,
        w_range=w_range,
        b_range=b_range,
        km_normalizer=km_normalizer,
        price_normalizer=price_normalizer,
    )

    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    return VisualizationState(fig=fig, axs=axs, data=plot_data)


def create_plot_callback(
    vis: VisualizationState,
) -> Callable[[TrainingStep], None]:
    """
    Factory function that creates and returns the update_plot callback.
    It encapsulates the visualization state so the model doesn't need to know about it.
    """

    def update_plot(step_data: TrainingStep) -> None:
        d = vis.data

        w, b = step_data.w, step_data.b
        grad_w, grad_b = step_data.grad_w, step_data.grad_b
        step = step_data.step

        current_preds = w * d.x + b
        residuals = current_preds - d.y
        current_sse = np.sum(residuals**2)

        for ax in vis.axs.flat:
            ax.clear()

        # --- TOP-LEFT: Line Fit (ORIGINAL SCALE) ---
        vis.axs[0, 0].scatter(d.x_raw, d.y_raw, color="blue", label="Real Data")

        # Create a line of raw X values spanning the min and max of the dataset
        x_line_raw = np.linspace(min(d.x_raw), max(d.x_raw), 100)

        # De-normalize process for the prediction line
        x_line_norm = d.km_normalizer.transform(x_line_raw)
        y_line_norm = w * x_line_norm + b
        y_line_raw = d.price_normalizer.inverse_transform(y_line_norm)

        vis.axs[0, 0].plot(x_line_raw, y_line_raw, color="red", label="Prediction")
        vis.axs[0, 0].set_title(
            f"Iter: {step} | SSE (norm): {current_sse:.2f}\nReal Scale Fit"
        )
        vis.axs[0, 0].set_xlabel("Mileage (km)")
        vis.axs[0, 0].set_ylabel("Price (€)")
        vis.axs[0, 0].legend()

        # --- TOP-RIGHT: Contour ---
        vis.axs[0, 1].contourf(d.w, d.b, d.z, levels=30, cmap="viridis")
        vis.axs[0, 1].scatter(
            w, b, c="red", s=80, edgecolors="white", label="Current (w, b)"
        )
        vis.axs[0, 1].set_title("Cost Landscape (Normalized Space)")
        vis.axs[0, 1].set_xlabel("w")
        vis.axs[0, 1].set_ylabel("b")
        vis.axs[0, 1].legend()

        # --- BOTTOM-LEFT: Cost vs W ---
        sse_w = [np.sum(((w_val * d.x + b) - d.y) ** 2) for w_val in d.w_range]
        vis.axs[1, 0].plot(d.w_range, sse_w, color="green")
        vis.axs[1, 0].scatter(w, current_sse, c="red", s=80, zorder=10)
        tangent_w = grad_w * (d.w_range - w) + current_sse
        vis.axs[1, 0].plot(d.w_range, tangent_w, color="orange", linestyle="--")
        vis.axs[1, 0].set_title("Cost vs w")

        # --- BOTTOM-RIGHT: Cost vs B ---
        sse_b = [np.sum(((w * d.x + b_val) - d.y) ** 2) for b_val in d.b_range]
        vis.axs[1, 1].plot(d.b_range, sse_b, color="purple")
        vis.axs[1, 1].scatter(b, current_sse, c="red", s=80, zorder=10)
        tangent_b = grad_b * (d.b_range - b) + current_sse
        vis.axs[1, 1].plot(d.b_range, tangent_b, color="orange", linestyle="--")
        vis.axs[1, 1].set_title("Cost vs b")

        plt.pause(0.01)

    return update_plot
