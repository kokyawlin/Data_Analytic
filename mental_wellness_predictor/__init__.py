"""
Mental Wellness Prediction Model Package

This package provides tools and models for predicting mental wellness
situations based on various behavioral and psychological indicators.
"""

__version__ = "1.0.0"
__author__ = "Mental Wellness Analytics Team"

from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .predictor import WellnessPredictor

__all__ = ['DataProcessor', 'FeatureEngineer', 'ModelTrainer', 'WellnessPredictor']