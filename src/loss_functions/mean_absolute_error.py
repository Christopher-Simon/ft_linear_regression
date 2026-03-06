from collections.abc import Callable

from src.loss_functions.protocol_loss_fn import LossFunction


class MeanAbsoluteError(LossFunction):
    """
    Class to calculate the Mean Absolute Error (MAE) loss function.
    Formula: J = 1/m * sum(|pred - y|)
    """

    def _sign(self, x: float) -> float:
        if x == 0:
            return 0.0
        elif x < 0:
            return -1.0
        else:
            return 1.0

    def loss(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        Calculate the Mean Absolute Error cost.
        """
        total_error: float = 0
        m = len(x_list)
        for x, y in zip(x_list, y_list):
            total_error += abs(estimate_func(x) - y)
        return total_error / m

    def derived_b(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        Derivation with respect to b: 1/m * sum(sgn(pred - y))
        """
        total_sum: float = 0
        m = len(x_list)
        for x, y in zip(x_list, y_list):
            res = self._sign(estimate_func(x) - y)
            total_sum += res
        return total_sum / m

    def derived_w(
        self,
        x_list: list[float],
        y_list: list[float],
        estimate_func: Callable[[float], float],
    ) -> float:
        """
        Derivation with respect to w: 1/m * sum(sgn(pred - y) * x)
        """
        total_sum: float = 0
        m = len(x_list)
        for x, y in zip(x_list, y_list):
            res = self._sign(estimate_func(x) - y) * x
            total_sum += res
        return total_sum / m
