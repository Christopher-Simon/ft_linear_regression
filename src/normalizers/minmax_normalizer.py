"""
Min-Max Normalization
"""

import numpy as np
from pydantic import BaseModel

from src.normalizers.protocol_normalizers import Normalizer


class MinMaxParams(BaseModel):
    min: float = 0.0
    max: float = 1.0
    range: float = 1.0


class MinMaxNormalizer(Normalizer[MinMaxParams]):
    def __init__(self, params: MinMaxParams | None = None) -> None:
        self.params = params if params is not None else MinMaxParams()

    def fit(self, x: np.ndarray) -> None:
        if x.size == 0:
            raise ValueError("Input array is empty")

        self.params.min = float(np.min(x))
        self.params.max = float(np.max(x))
        self.params.range = self.params.max - self.params.min

        if self.params.range == 0:
            self.params.range = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.params.min) / self.params.range

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.params.range) + self.params.min
