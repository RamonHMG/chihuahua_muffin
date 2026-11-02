"""SPEC 02 model, training, and evaluation implementation."""

from .architecture import EXPECTED_TRAINABLE_PARAMETERS, build_model
from .config import ModelRunConfig, load_model_config
from .contract import DatasetContract, DatasetSample, load_dataset_contract

__all__ = [
    "DatasetContract",
    "DatasetSample",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "ModelRunConfig",
    "build_model",
    "load_dataset_contract",
    "load_model_config",
]
