"""
This module contains the implementation of different loss functions.
"""

from typing import Protocol
import numpy as np


class LossFunction(Protocol):
    """
    Interface for all LossFunctions
    """

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss between the true and predicted values
        """
        ...

    def compute_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray
    ) -> tuple:
        """
        Compute the gradient of the loss function
        """
        ...


# Implement different loss functions
class MeanAbsoluteError:
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_pred - y_true))

    def compute_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray
    ) -> tuple:
        error = np.sign(y_pred - y_true)
        d_theta_0 = np.mean(error)
        d_theta_1 = np.mean(error * X)
        return d_theta_0, d_theta_1


class MeanBiasError:
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_pred - y_true)

    def compute_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray
    ) -> tuple:
        error = y_pred - y_true
        d_theta_0 = np.mean(error)
        d_theta_1 = np.mean(error * X)
        return d_theta_0, d_theta_1


class MeanSquaredError:
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)

    def compute_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray
    ) -> tuple:
        error = y_pred - y_true
        d_theta_0 = np.mean(error)
        d_theta_1 = np.mean(error * X)
        return d_theta_0, d_theta_1
