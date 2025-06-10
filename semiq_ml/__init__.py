"""
Semiq-ML helper Package

This package provides helper functions and classes to simplify common machine learning workflows,
including baseline model training, hyperparameter tuning, and image processing.
"""

__version__ = "0.3.5"

# Import main classes for easy access
from .baseline_model import BaselineModel
from .tuning import OptunaOptimizer
import semiq_ml.image

# Define what's available when using "from ml_helper import *"
__all__ = [
    'BaselineModel',
    'OptunaOptimizer',
    'image',
]