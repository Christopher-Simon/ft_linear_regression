import os
import sys

import numpy as np
import pandas as pd


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
