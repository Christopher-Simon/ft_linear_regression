"""
Main module for the application.
"""

import json
import os
import sys
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluator.r_squared import r_squared
from model.simple_linear_regression import SimpleLinearRegression
from normalizers.minmax_normalizer import MinMaxNormalizer, MinMaxParams
from normalizers.protocol_normalizers import Normalizer
from normalizers.z_score import ZScoreParams
from visualisation.data_graph import create_plot_callback, init_visualization


class NormalizerType(StrEnum):
    ZSCORE = "zscore"
    MINMAX = "minmax"


def get_normalizer_name(
    normalizer: Normalizer[ZScoreParams] | Normalizer[MinMaxParams],
) -> str:
    """Returns the enum string based on the normalizer instance."""
    if isinstance(normalizer, MinMaxNormalizer):
        return NormalizerType.MINMAX.value
    return NormalizerType.ZSCORE.value


def load_and_clean_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Safely loads the CSV, handles missing values, removes invalid data types,
    and ensures there is enough data to perform linear regression.
    """
    if not os.path.exists(filepath):
        print(f"Error: Dataset '{filepath}' not found.")
        sys.exit(1)

    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error: Could not read the CSV file. Details: {e}")
        sys.exit(1)

    if "km" not in data.columns or "price" not in data.columns:
        print("Error: The dataset must contain 'km' and 'price' columns.")
        sys.exit(1)

    data["km"] = pd.to_numeric(data["km"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    cleaned_data = data.dropna(subset=["km", "price"])

    if len(cleaned_data) < 2:
        print(
            f"Error: Insufficient valid data. Found {len(cleaned_data)} valid rows, "
            "but at least 2 are required to calculate a regression line."
        )
        sys.exit(1)

    x_raw = cleaned_data["km"].to_numpy(dtype=float)
    y_raw = cleaned_data["price"].to_numpy(dtype=float)

    return x_raw, y_raw


def train(dataset_path: str = "data/data.csv") -> None:
    """
    Main orchestrator for data loading, training, and saving.
    """
    x_raw, y_raw = load_and_clean_data(dataset_path)

    km_normalizer = MinMaxNormalizer()
    price_normalizer = MinMaxNormalizer()
    x = km_normalizer.fit_transform(x_raw)
    y = price_normalizer.fit_transform(y_raw)

    vis_state = init_visualization(
        x=x,
        y=y,
        x_raw=x_raw,
        y_raw=y_raw,
        km_normalizer=km_normalizer,
        price_normalizer=price_normalizer,
    )
    plot_callback = create_plot_callback(vis_state)

    custom_model = SimpleLinearRegression()
    custom_model.fit(
        x_list=x.tolist(),
        y_list=y.tolist(),
        learning_rate=0.1,
        iterations=100000,
        callback=plot_callback,
    )

    plt.ioff()

    precision_r2 = r_squared(
        x_list=x.tolist(),
        y_true=y.tolist(),
        line_params=(custom_model.slope, custom_model.intercept),
    )

    final_loss = custom_model.loss_fn.loss(
        x_list=x.tolist(),
        y_list=y.tolist(),
        estimate_func=custom_model.estimate_price,
    )

    model_data = {
        "slope": custom_model.slope,
        "intercept": custom_model.intercept,
        "normalizer_type": get_normalizer_name(km_normalizer),
        "km_params": km_normalizer.params.model_dump(),
        "price_params": price_normalizer.params.model_dump(),
        "precision_r2": precision_r2,
        "final_loss": final_loss,
    }

    with open("model/model_weights.json", "w") as f:
        json.dump(model_data, f, indent=4)

    print("==================================================")
    print("Training complete! Weights saved to 'model_weights.json'.")
    print(f"Final Model Loss (Normalized) : {final_loss:.6f}")
    print(
        f"Model Precision (R-squared)   : \
            {precision_r2:.4f} ({(precision_r2 * 100):.2f}%)"
    )
    print("==================================================")

    plt.show()


if __name__ == "__main__":
    # If the user provides a command-line argument, use it as the path.
    # e.g., `python src/train.py data/custom_data.csv`
    custom_path = sys.argv[1] if len(sys.argv) > 1 else "data/data.csv"
    train(custom_path)
