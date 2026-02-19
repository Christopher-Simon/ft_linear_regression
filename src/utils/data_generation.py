import os

import numpy as np
import pandas as pd


def generate_datasets() -> None:
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)  # For reproducible results

    # ---------------------------------------------------------
    # Dataset 1: Standard Depreciation (Realistic)
    # Equation: Price = 25000 - 0.07 * km + noise
    # Tests a normal, expected car depreciation curve.
    # ---------------------------------------------------------
    km1 = np.random.uniform(0, 250000, 50)
    price1 = 25000 - 0.07 * km1 + np.random.normal(0, 1500, 50)
    pd.DataFrame(
        {"km": km1.astype(int), "price": np.maximum(price1, 1000).astype(int)}
    ).to_csv("data/valid_1_standard.csv", index=False)

    # ---------------------------------------------------------
    # Dataset 2: Positive Correlation (Mathematical Edge Case)
    # Equation: Price = 5000 + 0.15 * km + noise
    # Cars usually depreciate, but mathematically your model
    # should be able to learn a positive weight just as easily!
    # ---------------------------------------------------------
    km2 = np.random.uniform(10000, 150000, 100)
    price2 = 5000 + 0.15 * km2 + np.random.normal(0, 1000, 100)
    pd.DataFrame({"km": km2.astype(int), "price": price2.astype(int)}).to_csv(
        "data/valid_2_positive.csv", index=False
    )

    # ---------------------------------------------------------
    # Dataset 3: High Variance (Noisy Data)
    # Equation: Price = 20000 - 0.05 * km + HEAVY noise
    # Tests if your model can find the true line of best fit
    # even when the data points are scattered chaotically.
    # ---------------------------------------------------------
    km3 = np.random.uniform(0, 300000, 75)
    price3 = 20000 - 0.05 * km3 + np.random.normal(0, 5000, 75)
    pd.DataFrame(
        {"km": km3.astype(int), "price": np.maximum(price3, 500).astype(int)}
    ).to_csv("data/valid_3_noisy.csv", index=False)

    # ---------------------------------------------------------
    # Dataset 4: The Giant (2000 rows)
    # Equation: Price = 35000 - 0.08 * km + noise
    # Tests the performance and stability of your gradient
    # descent loop over a massive amount of data.
    # ---------------------------------------------------------
    km4 = np.random.uniform(0, 400000, 2000)
    price4 = 35000 - 0.08 * km4 + np.random.normal(0, 2000, 2000)
    pd.DataFrame(
        {"km": km4.astype(int), "price": np.maximum(price4, 1000).astype(int)}
    ).to_csv("data/valid_4_large.csv", index=False)

    print("✅ Successfully generated 4 test datasets in the 'data/' folder!")
    print("  - data/valid_1_standard.csv (50 rows)")
    print("  - data/valid_2_positive.csv (100 rows)")
    print("  - data/valid_3_noisy.csv    (75 rows)")
    print("  - data/valid_4_large.csv    (2000 rows)")


if __name__ == "__main__":
    generate_datasets()
