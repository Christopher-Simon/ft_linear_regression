"""
Evaluate module to calculate the precision of the trained algorithm.
"""

import sys

from src.evaluator.r_squared import r_squared
from src.utils.read_model import get_normalizer, init
from train import load_and_clean_data


def evaluate(dataset_path: str, model_file: str) -> None:
    print("=========================================")
    print("       Algorithm Precision Evaluator     ")
    print("=========================================")

    print(f"Loading dataset: '{dataset_path}'...")
    x_raw, y_raw = load_and_clean_data(dataset_path)

    print(f"Loading model weights: '{model_file}'...")
    model_params = init(model_file)

    if model_params.slope == 0.0 and model_params.intercept == 0.0:
        print("\nWarning: The model weights are 0.0. Did you forget to train it?")

    km_normalizer = get_normalizer(model_params.norm_type, model_params.km_params_dict)
    price_normalizer = get_normalizer(
        model_params.norm_type, model_params.price_params_dict
    )

    x = km_normalizer.transform(x_raw)
    y = price_normalizer.transform(y_raw)
    print(
        f"Loaded Weights -> Slope: {model_params.slope},\
            Intercept: {model_params.intercept}"
    )

    precision_r2 = r_squared(
        x_list=x.tolist(),
        y_true=y.tolist(),
        line_params=(model_params.slope, model_params.intercept),
    )

    print("\n--- Precision Metrics ---")
    print(f"Data Points Evaluated : {len(x_raw)}")
    print(f"R-squared Score       : {precision_r2:.4f} ({precision_r2 * 100:.2f}%)")
    print("=========================================")


if __name__ == "__main__":
    custom_data = sys.argv[1] if len(sys.argv) > 1 else "data/data.csv"
    custom_model = sys.argv[2] if len(sys.argv) > 2 else "model/model_weights.json"
    evaluate(custom_data, custom_model)
