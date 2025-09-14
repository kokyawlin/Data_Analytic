"""
Data Processing Module for Mental Wellness Prediction

This module handles data loading, cleaning, and preprocessing for mental wellness analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data preprocessing for mental wellness prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.target_column = None
    
    def load_data(self, file_path=None, data=None):
        """Load data from file or use provided data."""
        if data is not None:
            self.data = data
        elif file_path:
            try:
                self.data = pd.read_csv(file_path)
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
        else:
            # Generate sample data if no file provided
            self.data = self.generate_sample_data()
        
        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        return self.data
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic mental wellness data for demonstration."""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'sleep_hours': np.random.normal(7, 1.5, n_samples),
            'exercise_frequency': np.random.randint(0, 7, n_samples),
            'social_interaction_score': np.random.randint(1, 10, n_samples),
            'work_stress_level': np.random.randint(1, 10, n_samples),
            'screen_time_hours': np.random.normal(6, 2, n_samples),
            'financial_stress': np.random.randint(1, 10, n_samples),
            'relationship_satisfaction': np.random.randint(1, 10, n_samples),
            'physical_health_score': np.random.randint(1, 10, n_samples),
            'substance_use_frequency': np.random.randint(0, 5, n_samples),
            'meditation_frequency': np.random.randint(0, 7, n_samples),
            'therapy_sessions': np.random.randint(0, 20, n_samples),
        }
        
        # Create target variable based on features
        wellness_score = (
            (data['sleep_hours'] - 7) * 2 +
            data['exercise_frequency'] * 1.5 +
            data['social_interaction_score'] * 1.2 +
            (10 - data['work_stress_level']) * 1.8 +
            (10 - data['financial_stress']) * 1.3 +
            data['relationship_satisfaction'] * 1.5 +
            data['physical_health_score'] * 1.1 +
            (5 - data['substance_use_frequency']) * 2 +
            data['meditation_frequency'] * 1.2 +
            data['therapy_sessions'] * 0.5 +
            np.random.normal(0, 5, n_samples)  # Add noise
        )
        
        # Normalize and categorize wellness
        wellness_score = (wellness_score - wellness_score.min()) / (wellness_score.max() - wellness_score.min())
        data['mental_wellness_category'] = pd.cut(
            wellness_score,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Poor', 'Moderate', 'Good'],
            include_lowest=True
        )
        
        return pd.DataFrame(data)
    
    def clean_data(self):
        """Clean and preprocess the data."""
        # Handle missing values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.imputer.fit_transform(self.data[numeric_columns])
        
        # Remove outliers (using IQR method)
        for col in numeric_columns:
            if col != 'age':  # Don't remove age outliers
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[
                    (self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)
                ]
        
        # Ensure sleep hours are reasonable
        self.data['sleep_hours'] = self.data['sleep_hours'].clip(3, 12)
        
        # Ensure screen time is reasonable
        self.data['screen_time_hours'] = self.data['screen_time_hours'].clip(0, 16)
        
        logger.info(f"Data cleaned. Final shape: {self.data.shape}")
        return self.data
    
    def prepare_features_target(self, target_column='mental_wellness_category'):
        """Separate features and target variable."""
        self.target_column = target_column
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Encode categorical target if needed
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)
            # Convert to numpy array to ensure integer type
            y = np.array(y, dtype=int)
        
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Features prepared. Shape: {X.shape}")
        logger.info(f"Target classes: {self.label_encoder.classes_ if hasattr(self.label_encoder, 'classes_') else np.unique(y)}")
        logger.info(f"Target data type: {type(y)}, unique values: {np.unique(y)}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_names(self):
        """Get the names of features."""
        return self.feature_columns
    
    def get_target_classes(self):
        """Get the target class names."""
        if hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.classes_
        return None