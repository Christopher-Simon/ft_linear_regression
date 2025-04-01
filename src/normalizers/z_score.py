"""
Z-score Normalization
"""

import pandas as pd
from normalizers.protocol_normalizers import Normalizer


class ZScoreNormalizer(Normalizer):
    """
    Z-score normalization for pandas DataFrames:
        z = (x - mean) / std
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data
        self.normalized_data: pd.DataFrame = data.copy()

    def get_main_values(self) -> None:
        """
        Get the mean, deviation, min and max values of the DataFrame.
        """
        # print mean, std, min and max values of the km column
        print("km mean: ", self.data["km"].mean)
        print("km std: ", self.data["km"].std)
        print("km min: ", self.data["km"].min)
        print("km max: ", self.data["km"].max)
        # print mean, std, min and max values of the price column
        print("price mean: ", self.data["price"].mean)
        print("price std: ", self.data["price"].std)
        print("price min: ", self.data["price"].min)
        print("price max: ", self.data["price"].max)

    def transform(self) -> pd.DataFrame:
        """
        Transform the data using the learned parameters. This method should
        apply the normalization to the data.
        """
        self.normalized_data["km"] = (
            self.data["km"] - self.data["km"].mean()
        ) / self.data["km"].std()
        self.normalized_data["price"] = (
            self.data["price"] - self.data["price"].mean()
        ) / self.data["price"].std()
        return self.normalized_data

    def invert_km(self, km: float) -> float:
        """
        invert the km to its normalized value.
        """
        return (km - self.data["km"].mean()) / self.data["km"].std()

    def invert_price(self, price: float) -> float:
        """
        invert the price to its normalized value.
        """
        return (price - self.data["price"].mean()) / self.data["price"].std()

    def invert_weight_bias(
        self,
        weight: float,
        bias: float,
    ) -> tuple[float, float]:
        """
        invert the weight and bias to its normalized value.
        """
        return (
            weight * (self.data["price"].std() / self.data["km"].std()),
            self.data["price"].mean()
            + self.data["price"].std() * bias
            - (
                self.data["price"].std()
                / self.data["km"].std()
                * self.data["km"].mean()
                * weight
            ),
        )
