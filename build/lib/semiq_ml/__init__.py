"""
Semiq-ML helper Package

This package provides helper functions and classes to simplify common machine learning workflows,
including baseline model training and hyperparameter tuning.
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .baseline_model import BaselineModel
from .tuning import GridSearchOptimizer, RandomSearchOptimizer
# from .hyperparameter_tuning import RandomSearchOptimizer

# Define what's available when using "from ml_helper import *"
__all__ = [
    'BaselineModel',
    'RandomSearchOptimizer',
    'GridSearchOptimizer',
]