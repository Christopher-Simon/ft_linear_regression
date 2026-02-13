"""
Predict module for the application.
"""

import json
import os
import sys

import numpy as np

from model.simple_linear_regression import SimpleLinearRegression
from normalizers.z_score import ZScoreNormalizer, ZScoreParams


def predict() -> None:
    """
    Prediction program:
    Reads from model_weights.json, prompts for a mileage,
    normalizes the input, predicts, and inverse_transforms the result.
    """
    model_file = "model/model_weights.json"

    slope = 0.0
    intercept = 0.0
    km_params = ZScoreParams()
    price_params = ZScoreParams()
    precision = None

    if os.path.exists(model_file):
        try:
            with open(model_file) as f:
                data = json.load(f)
                slope = data.get("slope", 0.0)
                intercept = data.get("intercept", 0.0)

                km_params = ZScoreParams(
                    mean=data.get("km_mean", 0.0), std=data.get("km_std", 1.0)
                )
                price_params = ZScoreParams(
                    mean=data.get("price_mean", 0.0), std=data.get("price_std", 1.0)
                )
                precision = data.get("precision_r2")

        except json.JSONDecodeError:
            print("Error: model_weights.json is corrupted. Using default weights (0).")

    model = SimpleLinearRegression(slope=slope, intercept=intercept)
    km_normalizer = ZScoreNormalizer(km_params)
    price_normalizer = ZScoreNormalizer(price_params)

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

            price_norm = model.estimate_price(km_norm)

            price_raw = price_normalizer.inverse_transform(np.array([price_norm]))[0]

            estimated_price = max(0.0, float(price_raw))

            print(f"-> Estimated price for {mileage:,.2f} km: {estimated_price:,.2f} €")

            if precision is not None:
                print(f"   (Model Confidence / R²: {precision * 100:.2f}%)")

        except ValueError:
            print("Invalid input. Please enter a numerical value.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    predict()
