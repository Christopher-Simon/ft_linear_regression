"""
Main module for the application.
"""

import json
import sys

import matplotlib.pyplot as plt

from src.evaluator.r_squared import r_squared
from src.model.simple_linear_regression import SimpleLinearRegression
from src.normalizers.z_score import ZScoreNormalizer
from src.utils.load import load_and_clean_data
from src.utils.read_model import get_normalizer_name
from src.visualisation.data_graph import create_plot_callback, init_visualization


def train(dataset_path: str) -> None:
    """
    Main orchestrator for data loading, training, and saving.
    """
    x_raw, y_raw = load_and_clean_data(dataset_path)

    km_normalizer = ZScoreNormalizer()
    price_normalizer = ZScoreNormalizer()
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
    custom_path = sys.argv[1] if len(sys.argv) > 1 else "data/data.csv"
    train(custom_path)
