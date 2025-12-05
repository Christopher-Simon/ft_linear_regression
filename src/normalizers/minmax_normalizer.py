"""
Min-Max Normalization"""

import pandas as pd
from normalizers.protocol_normalizers import Normalizer


class MinMaxNormalizer(Normalizer):
    """
    Min-Max normalization for pandas DataFrames:
        x_norm = (x - min) / (max - min)
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data
        self.normalized_data: pd.DataFrame = data.copy()

    def get_main_values(self) -> None:
        """
        Get the mean, deviation, min and max values of the DataFrame.
        """
        print("km mean: ", self.data["km"].mean)
        print("km std: ", self.data["km"].std)
        print("km min: ", self.data["km"].min)
        print("km max: ", self.data["km"].max)
        print("price mean: ", self.data["price"].mean)
        print("price std: ", self.data["price"].std)
        print("price min: ", self.data["price"].min)
        print("price max: ", self.data["price"].max)

    def transform(self) -> pd.DataFrame:
        """
        Transform the data using the learned parameters. This method should
        apply the normalization to the data.
        """
        self.normalized_data["km"] = (self.data["km"] - self.data["km"].min()) / (
            self.data["km"].max() - self.data["km"].min()
        )
        self.normalized_data["price"] = (
            self.data["price"] - self.data["price"].min()
        ) / (self.data["price"].max() - self.data["price"].min())
        return self.normalized_data

    def invert_km(self, km: float) -> float:
        """
        invert the km to its normalized value.
        """
        return (
            km * (self.data["km"].max() - self.data["km"].min()) + self.data["km"].min()
        )

    def invert_price(self, price: float) -> float:
        """
        invert the price to its normalized value.
        """
        return (
            price * (self.data["price"].max() - self.data["price"].min())
            + self.data["price"].min()
        )

    def invert_weight_bias(
        self,
        weight: float,
        bias: float,
    ) -> tuple[float, float]:
        """
        invert the weight and bias to its normalized value.
        """

        original_weight = (
            weight
            * (self.data["price"].max() - self.data["price"].min())
            / (self.data["km"].max() - self.data["km"].min())
        )

        return (
            original_weight,
            self.data["price"].min()
            - original_weight * self.data["km"].min()
            + (self.data["price"].max() - self.data["price"].min()) * bias,
        )
