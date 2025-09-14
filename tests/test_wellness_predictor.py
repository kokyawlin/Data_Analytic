"""
Unit Tests for Mental Wellness Predictor

Basic tests to ensure the components work correctly.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mental_wellness_predictor.data_processor import DataProcessor
from mental_wellness_predictor.feature_engineer import FeatureEngineer
from mental_wellness_predictor.model_trainer import ModelTrainer
from mental_wellness_predictor import WellnessPredictor


class TestDataProcessor(unittest.TestCase):
    """Test the DataProcessor class."""
    
    def setUp(self):
        self.processor = DataProcessor()
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        data = self.processor.generate_sample_data(n_samples=100)
        
        self.assertEqual(len(data), 100)
        self.assertIn('mental_wellness_category', data.columns)
        self.assertIn('age', data.columns)
        self.assertIn('sleep_hours', data.columns)
        
        # Check data types and ranges
        self.assertTrue(data['age'].min() >= 18)
        self.assertTrue(data['age'].max() <= 80)
        self.assertTrue(data['sleep_hours'].min() >= 0)
    
    def test_load_data(self):
        """Test data loading functionality."""
        data = self.processor.load_data()  # Should generate sample data
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
    
    def test_clean_data(self):
        """Test data cleaning."""
        self.processor.load_data()
        cleaned_data = self.processor.clean_data()
        
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertGreater(len(cleaned_data), 0)
        
        # Check that sleep hours are within reasonable range
        self.assertTrue(cleaned_data['sleep_hours'].min() >= 3)
        self.assertTrue(cleaned_data['sleep_hours'].max() <= 12)
    
    def test_prepare_features_target(self):
        """Test feature and target preparation."""
        self.processor.load_data()
        self.processor.clean_data()
        
        X, y = self.processor.prepare_features_target()
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, (np.ndarray, pd.Series))
        self.assertEqual(len(X), len(y))
        self.assertNotIn('mental_wellness_category', X.columns)


class TestFeatureEngineer(unittest.TestCase):
    """Test the FeatureEngineer class."""
    
    def setUp(self):
        self.engineer = FeatureEngineer()
        self.processor = DataProcessor()
        self.processor.load_data()
        self.processor.clean_data()
        self.X, self.y = self.processor.prepare_features_target()
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        X_enhanced = self.engineer.create_interaction_features(self.X)
        
        self.assertGreater(len(X_enhanced.columns), len(self.X.columns))
        self.assertEqual(len(X_enhanced), len(self.X))
    
    def test_create_risk_scores(self):
        """Test risk score creation."""
        X_risk = self.engineer.create_risk_scores(self.X)
        
        self.assertGreater(len(X_risk.columns), len(self.X.columns))
        self.assertIn('overall_risk_score', X_risk.columns)
    
    def test_engineer_all_features(self):
        """Test comprehensive feature engineering."""
        X_engineered = self.engineer.engineer_all_features(self.X, self.y)
        
        self.assertIsInstance(X_engineered, pd.DataFrame)
        self.assertEqual(len(X_engineered), len(self.X))


class TestModelTrainer(unittest.TestCase):
    """Test the ModelTrainer class."""
    
    def setUp(self):
        self.trainer = ModelTrainer()
        self.processor = DataProcessor()
        self.processor.load_data()
        self.processor.clean_data()
        self.X, self.y = self.processor.prepare_features_target()
    
    def test_initialize_models(self):
        """Test model initialization."""
        self.trainer.initialize_models()
        
        self.assertGreater(len(self.trainer.models), 0)
        self.assertIn('random_forest', self.trainer.models)
        self.assertIn('gradient_boosting', self.trainer.models)
    
    def test_handle_imbalanced_data(self):
        """Test SMOTE balancing."""
        X_balanced, y_balanced = self.trainer.handle_imbalanced_data(self.X, self.y)
        
        self.assertIsInstance(X_balanced, (pd.DataFrame, np.ndarray))
        self.assertIsInstance(y_balanced, np.ndarray)
        self.assertEqual(len(X_balanced), len(y_balanced))
    
    def test_cross_validate_models(self):
        """Test cross-validation (with small dataset)."""
        # Use a small subset for faster testing
        X_small = self.X.head(50)
        y_small = self.y[:50]
        
        self.trainer.cross_validate_models(X_small, y_small, cv_folds=3)
        
        self.assertGreater(len(self.trainer.cv_results), 0)
        self.assertIsNotNone(self.trainer.best_model_name)


class TestWellnessPredictor(unittest.TestCase):
    """Test the main WellnessPredictor class."""
    
    def setUp(self):
        self.predictor = WellnessPredictor()
    
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertIsNotNone(self.predictor.data_processor)
        self.assertIsNotNone(self.predictor.feature_engineer)
        self.assertIsNotNone(self.predictor.model_trainer)
        self.assertFalse(self.predictor.is_trained)
    
    def test_train_model(self):
        """Test model training (quick version)."""
        # Train with minimal parameters for speed
        results = self.predictor.train_model(
            use_feature_engineering=False,
            tune_hyperparameters=False
        )
        
        self.assertTrue(self.predictor.is_trained)
        self.assertIn('evaluation_results', results)
        self.assertIn('best_model_name', results)
    
    def test_predict_after_training(self):
        """Test predictions after training."""
        # First train the model
        self.predictor.train_model(
            use_feature_engineering=False,
            tune_hyperparameters=False
        )
        
        # Test single prediction
        sample_input = {
            'age': 30,
            'sleep_hours': 7.0,
            'exercise_frequency': 3,
            'social_interaction_score': 6,
            'work_stress_level': 5,
            'screen_time_hours': 6.0,
            'financial_stress': 4,
            'relationship_satisfaction': 7,
            'physical_health_score': 7,
            'substance_use_frequency': 1,
            'meditation_frequency': 2,
            'therapy_sessions': 0
        }
        
        result = self.predictor.predict_single(**sample_input)
        
        self.assertIn('predicted_wellness', result)
        self.assertIn('confidence_scores', result)
        self.assertIn('risk_assessment', result)
        self.assertIn('recommendations', result)


class TestDataTypes(unittest.TestCase):
    """Test data types and validation."""
    
    def test_synthetic_data_quality(self):
        """Test quality of synthetic data generation."""
        processor = DataProcessor()
        data = processor.generate_sample_data(n_samples=200)
        
        # Check for reasonable distributions
        self.assertGreater(data['sleep_hours'].std(), 0.5)  # Should have variation
        self.assertLess(data['sleep_hours'].std(), 3.0)     # But not too much
        
        # Check categorical target
        categories = data['mental_wellness_category'].unique()
        self.assertGreaterEqual(len(categories), 2)  # Should have multiple categories
        
        # Check no extreme outliers in key features
        for col in ['exercise_frequency', 'social_interaction_score', 'work_stress_level']:
            self.assertTrue(data[col].min() >= 0)
            self.assertTrue(data[col].max() <= 10)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)