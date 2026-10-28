"""Build selection models and shared model infrastructure."""

from .baseline_models import NoHardwareInteractionModel, NoHardwareModel, RandomBaseline
from .common import BaseModel
from .ridge_regression import PerformanceModel
from .ridge_regression_interactions import InteractionPerformanceModel
from .ridge_regression_three_way import ThreeWayPerformanceModel
from .random_forest import RandomForestPerformanceModel
from .catboost import CatBoostPerformanceModel
from .gaussian_process import GaussianProcessPerformanceModel
from .kernel_ridge import KernelRidgePerformanceModel
from .support_vector import SupportVectorPerformanceModel
from .small_mlp import SmallMLPPerformanceModel

# Public CLI names mapped to model constructors
# CatBoost wrapper import without loading the external dependency
MODEL_TYPES = {
    "hardware": PerformanceModel,
    "hardware-interactions": InteractionPerformanceModel,
    "no-hardware": NoHardwareModel,
    "no-hardware-interactions": NoHardwareInteractionModel,
    "hardware-three-way": ThreeWayPerformanceModel,
    "random": RandomBaseline,
    "random-forest": RandomForestPerformanceModel,
    "catboost": CatBoostPerformanceModel,
    "gaussian-process": GaussianProcessPerformanceModel,
    "kernel-ridge": KernelRidgePerformanceModel,
    "support-vector": SupportVectorPerformanceModel,
    "small-mlp": SmallMLPPerformanceModel,
}

# Public model classes for Python use and constructor tuning
__all__ = ["BaseModel", "PerformanceModel", "InteractionPerformanceModel", "ThreeWayPerformanceModel",
           "NoHardwareModel", "NoHardwareInteractionModel", "RandomBaseline",
           "RandomForestPerformanceModel", "CatBoostPerformanceModel",
           "GaussianProcessPerformanceModel", "KernelRidgePerformanceModel",
           "SupportVectorPerformanceModel", "SmallMLPPerformanceModel", "MODEL_TYPES"]
