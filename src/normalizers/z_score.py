"""
Z-score Normalization
"""

import numpy as np
from pydantic import BaseModel

from normalizers.protocol_normalizers import Normalizer


class ZScoreParams(BaseModel):
    mean: float = 0.0
    std: float = 1.0


class ZScoreNormalizer(Normalizer[ZScoreParams]):
    def __init__(self, params: ZScoreParams | None = None) -> None:
        self.params = params if params is not None else ZScoreParams()

    def fit(self, x: np.ndarray) -> None:
        if x.size == 0:
            raise ValueError("Input array is empty")

        self.params.mean = float(np.mean(x))
        std = float(np.std(x))

        if std == 0:
            std = 1.0
        self.params.std = std

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.params.mean) / self.params.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.params.std) + self.params.mean
