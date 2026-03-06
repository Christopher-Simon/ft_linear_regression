import os
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd


def load_and_clean_data(filepath: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Safely loads the CSV, handles missing values, removes invalid data types,
    and ensures there is enough data to perform linear regression.
    Works dynamically with ANY 2-column dataset.
    """
    if not os.path.exists(filepath):
        print(f"Error: Dataset '{filepath}' not found.")
        sys.exit(1)

    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error: Could not read the CSV file. Details: {e}")
        sys.exit(1)

    if len(data.columns) < 2:
        print("Error: The dataset must contain at least two columns (X and Y).")
        sys.exit(1)

    x_col = data.columns[0]
    y_col = data.columns[1]

    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")

    cleaned_data = data.dropna(subset=[x_col, y_col])

    if len(cleaned_data) < 2:
        print(
            f"Error: Insufficient valid data. Found {len(cleaned_data)} valid rows, "
            "but at least 2 are required to calculate a regression line."
        )
        sys.exit(1)

    x_raw = cleaned_data[x_col].to_numpy(dtype=float)
    y_raw = cleaned_data[y_col].to_numpy(dtype=float)

    return x_raw, y_raw