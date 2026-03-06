import json
import os
from dataclasses import dataclass
from enum import Enum

from src.normalizers.minmax_normalizer import MinMaxNormalizer, MinMaxParams
from src.normalizers.protocol_normalizers import Normalizer
from src.normalizers.z_score import ZScoreNormalizer, ZScoreParams


@dataclass
class ModelParams:
    """Dataclass to hold the extracted model parameters from JSON."""

    slope: float
    intercept: float
    precision: float | None
    norm_type: str
    km_params_dict: dict[str, float]
    price_params_dict: dict[str, float]


def init(model_file: str = "model_weights.json") -> ModelParams:
    """Reads the model parameters from a file and returns a ModelParams dataclass."""
    slope = 0.0
    intercept = 0.0
    precision = None

    norm_type = "zscore"
    km_params_dict = {}
    price_params_dict = {}

    if os.path.exists(model_file):
        try:
            with open(model_file) as f:
                data = json.load(f)
                slope = data.get("slope", 0.0)
                intercept = data.get("intercept", 0.0)
                precision = data.get("precision_r2")

                norm_type = data.get("normalizer_type", "zscore")
                km_params_dict = data.get("km_params", {})
                price_params_dict = data.get("price_params", {})

        except json.JSONDecodeError:
            print(f"Error: {model_file} is corrupted. Using default weights (0).")
    else:
        print(f"Info: '{model_file}' not found. Using default weights (0).")

    return ModelParams(
        slope=slope,
        intercept=intercept,
        precision=precision,
        norm_type=norm_type,
        km_params_dict=km_params_dict,
        price_params_dict=price_params_dict,
    )


class NormalizerType(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"


def get_normalizer_name(
    normalizer: Normalizer[ZScoreParams] | Normalizer[MinMaxParams],
) -> str:
    """Returns the enum string based on the normalizer instance."""
    if isinstance(normalizer, ZScoreNormalizer):
        return NormalizerType.ZSCORE.value
    return NormalizerType.MINMAX.value


def get_normalizer(
    name: str, params_dict: dict[str, float]
) -> Normalizer[ZScoreParams] | Normalizer[MinMaxParams]:
    """Factory function to dynamically reconstruct the correct normalizer."""
    if name == "minmax":
        return MinMaxNormalizer(MinMaxParams(**params_dict))

    return ZScoreNormalizer(ZScoreParams(**params_dict))
