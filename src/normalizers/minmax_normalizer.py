"""
Min-Max Normalization"""

import pandas as pd
from normalizers.protocol_normalizers import ColumnsMetrics, Normalizer


class MinMaxNormalizer(Normalizer):
    """
    Min-Max normalization for pandas DataFrames:
        x_norm = (x - min) / (max - min)
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data
        self.km_metrics = ColumnsMetrics(
            mean=self.data["km"].mean(),
            std=self.data["km"].std(),
            min_val=self.data["km"].min(),
            max_val=self.data["km"].max(),
        )
        self.price_metrics = ColumnsMetrics(
            mean=self.data["price"].mean(),
            std=self.data["price"].std(),
            min_val=self.data["price"].min(),
            max_val=self.data["price"].max(),
        )
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
        self.normalized_data["km"] = (self.data["km"] - self.km_metrics.min) / (
            self.km_metrics.max - self.km_metrics.min
        )
        self.normalized_data["price"] = (
            self.data["price"] - self.price_metrics.min
        ) / (self.price_metrics.max - self.price_metrics.min)
        return self.normalized_data

    def invert_km(self, km: float) -> float:
        """
        invert the km to its normalized value.
        """
        return km * (self.km_metrics.max - self.km_metrics.min) + self.km_metrics.min

    def invert_price(self, price: float) -> float:
        """
        invert the price to its normalized value.
        """
        return (
            price * (self.price_metrics.max - self.price_metrics.min)
            + self.price_metrics.min
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
            * (self.price_metrics.max - self.price_metrics.min)
            / (self.km_metrics.max - self.km_metrics.min)
        )

        return (
            original_weight,
            self.price_metrics.min
            - original_weight * self.km_metrics.min
            + (self.price_metrics.max - self.price_metrics.min) * bias,
        )
