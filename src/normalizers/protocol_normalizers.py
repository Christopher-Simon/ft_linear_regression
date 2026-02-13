"""
This module defines a protocol for normalizers in the context of data preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel

# Create a Type Variable that must be a Pydantic BaseModel (or subclass)
T = TypeVar("T", bound=BaseModel)


class Normalizer(ABC, Generic[T]):
    """
    Abstract base class to enforce structure without hardcoding column names.
    """

    # Tell the linter that params is of type T
    params: T

    def __init__(self, params: T | None = None) -> None:
        if params is not None:
            self.params = params

    @abstractmethod
    def fit(self, x: np.ndarray) -> None:
        """Calculate and store normalization parameters."""
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        pass

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and transform in one go."""
        self.fit(x)
        return self.transform(x)

    @abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Revert normalization."""
        pass
