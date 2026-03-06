"""
Predict module for the application.
"""

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

from src.model.simple_linear_regression import SimpleLinearRegression
from src.normalizers.minmax_normalizer import MinMaxNormalizer, MinMaxParams
from src.normalizers.protocol_normalizers import Normalizer
from src.normalizers.z_score import ZScoreNormalizer, ZScoreParams


@dataclass
class ModelParams:
    """Dataclass to hold the extracted model parameters from JSON."""

    slope: float
    intercept: float
    precision: float | None
    norm_type: str
    km_params_dict: dict[str, float]
    price_params_dict: dict[str, float]


def get_normalizer(
    name: str, params_dict: dict[str, float]
) -> Normalizer[ZScoreParams] | Normalizer[MinMaxParams]:
    """Factory function to dynamically reconstruct the correct normalizer."""
    print("params_dict", params_dict)
    if name == "minmax":
        return MinMaxNormalizer(MinMaxParams(**params_dict))

    return ZScoreNormalizer(ZScoreParams(**params_dict))


def init(model_file: str = "model_weights.json") -> ModelParams:
    """Reads the model parameters from a file and returns a ModelParams dataclass."""
    slope = 0.0
    intercept = 0.0
    precision = None

    norm_type = "zscore"
    km_params_dict = {}
    price_params_dict = {}

    if os.path.exists(model_file):
        try:
            with open(model_file) as f:
                data = json.load(f)
                slope = data.get("slope", 0.0)
                intercept = data.get("intercept", 0.0)
                precision = data.get("precision_r2")

                norm_type = data.get("normalizer_type", "zscore")
                km_params_dict = data.get("km_params", {})
                price_params_dict = data.get("price_params", {})

        except json.JSONDecodeError:
            print(f"Error: {model_file} is corrupted. Using default weights (0).")
    else:
        print(f"Info: '{model_file}' not found. Using default weights (0).")

    return ModelParams(
        slope=slope,
        intercept=intercept,
        precision=precision,
        norm_type=norm_type,
        km_params_dict=km_params_dict,
        price_params_dict=price_params_dict,
    )


def predict(model_file: str = "model_weights.json") -> None:
    model_params = init(model_file)

    model = SimpleLinearRegression(
        slope=model_params.slope, intercept=model_params.intercept
    )

    km_normalizer = get_normalizer(model_params.norm_type, model_params.km_params_dict)
    price_normalizer = get_normalizer(
        model_params.norm_type, model_params.price_params_dict
    )

    print("=========================================")
    print("        Car Price Predictor              ")
    print("=========================================")

    while True:
        try:
            user_input = input("\nPlease enter a mileage in km (or 'q' to quit): ")

            if user_input.strip().lower() in ["q", "quit", "exit"]:
                print("Goodbye!")
                break

            mileage = float(user_input)
            if mileage < 0:
                print("Mileage cannot be negative. Please try a valid number.")
                continue

            km_norm = km_normalizer.transform(np.array([mileage]))[0]
            print("km_norm", km_norm)

            price_norm = model.estimate_price(km_norm)
            print("price_norm", price_norm)
            price_raw = price_normalizer.inverse_transform(np.array([price_norm]))[0]

            estimated_price = max(0.0, float(price_raw))

            print(f"-> Estimated price for {mileage:,.2f} km: {estimated_price:,.2f} €")

            if model_params.precision is not None:
                print(
                    f"   (Model Confidence / R²: {model_params.precision * 100:.2f}%)"
                )

        except ValueError:
            print("Invalid input. Please enter a numerical value.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    custom_path = sys.argv[1] if len(sys.argv) > 1 else "model_weights.json"
    predict(custom_path)
