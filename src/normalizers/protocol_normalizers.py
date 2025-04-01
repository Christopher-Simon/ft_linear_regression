"""
This module defines a protocol for normalizers in the context of data preprocessing.
"""

from abc import abstractmethod
from typing import Protocol
import pandas as pd


class ColumnsMetrics:
    """
    A class to hold the metrics of columns in a DataFrame.
    It is used to store the mean, standard deviation, min, and max values of columns.
    """

    def __init__(self, mean: float, std: float, min_val: float, max_val: float) -> None:
        self.mean = mean
        self.std = std
        self.min = min_val
        self.max = max_val


class Normalizer(Protocol):
    """
    A protocol (interface-like) definition for normalizers.
    Classes implementing this protocol should provide:
      - a fit method to learn parameters (e.g., mean, std, min, max) from training data
      - a transform method to apply normalization to new data
    """

    data: pd.DataFrame
    normalized_data: pd.DataFrame

    def __init__(self, data_init: pd.DataFrame) -> None:
        """
        Initialize the normalizer with the data.
        :param data: The data to be normalized.
        """
        self.data = data_init
        self.normalized_data: pd.DataFrame = data_init.copy()

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        """
        Transform the data using the learned parameters. This method should
        apply the normalization to the data.
        """

    @abstractmethod
    def invert_km(self, km: float) -> float:
        """
        Invert the km to its normalized value.
        """

    @abstractmethod
    def invert_price(self, price: float) -> float:
        """
        Invert the price to its normalized value.
        """

    @abstractmethod
    def invert_weight_bias(
        self,
        weight: float,
        bias: float,
    ) -> tuple[float, float]:
        """
        Invert the weight and bias to their normalized values.
        """
