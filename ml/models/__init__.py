"""Build selection models and shared model infrastructure."""

from .baseline_models import (
    NoHardwareInteractionModel, NoHardwareModel, RandomBaseline,
    NoHardwareRandomForestModel, NoHardwareCatBoostModel,
    NoHardwareGaussianProcessModel, NoHardwareKernelRidgeModel,
    NoHardwareSupportVectorModel, NoHardwareSmallMLPModel,
)
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
from .smac_forest import SmacForestPerformanceModel
from .bayes_pairs import BayesPairsPerformanceModel
from .sparse_bayes import SparseBayesPerformanceModel
from .residual_correction import ResidualCorrection

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
    "smac-forest": SmacForestPerformanceModel,
    "bayes-pairs": BayesPairsPerformanceModel,
    "sparse-bayes": SparseBayesPerformanceModel,
    "no-hardware-random-forest": NoHardwareRandomForestModel,
    "no-hardware-catboost": NoHardwareCatBoostModel,
    "no-hardware-gaussian-process": NoHardwareGaussianProcessModel,
    "no-hardware-kernel-ridge": NoHardwareKernelRidgeModel,
    "no-hardware-support-vector": NoHardwareSupportVectorModel,
    "no-hardware-small-mlp": NoHardwareSmallMLPModel,
}

# Public model classes for Python use and constructor tuning
__all__ = ["BaseModel", "PerformanceModel", "InteractionPerformanceModel", "ThreeWayPerformanceModel",
           "NoHardwareModel", "NoHardwareInteractionModel", "RandomBaseline",
           "RandomForestPerformanceModel", "CatBoostPerformanceModel",
           "GaussianProcessPerformanceModel", "KernelRidgePerformanceModel",
           "SupportVectorPerformanceModel", "SmallMLPPerformanceModel",
           "SmacForestPerformanceModel", "BayesPairsPerformanceModel", "SparseBayesPerformanceModel",
           "ResidualCorrection", "NoHardwareRandomForestModel", "NoHardwareCatBoostModel",
           "NoHardwareGaussianProcessModel", "NoHardwareKernelRidgeModel",
           "NoHardwareSupportVectorModel", "NoHardwareSmallMLPModel", "MODEL_TYPES"]
