"""
Feature Engineering Module for Mental Wellness Prediction

This module creates additional features and performs feature selection for mental wellness analysis.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for mental wellness prediction."""
    
    def __init__(self):
        self.feature_selector = None
        self.poly_features = None
        self.selected_features = None
    
    def create_interaction_features(self, X):
        """Create interaction features between important variables."""
        X_enhanced = X.copy()
        
        # Sleep-related interactions
        if 'sleep_hours' in X.columns and 'work_stress_level' in X.columns:
            X_enhanced['sleep_stress_interaction'] = X['sleep_hours'] * (10 - X['work_stress_level'])
        
        # Exercise-social interaction
        if 'exercise_frequency' in X.columns and 'social_interaction_score' in X.columns:
            X_enhanced['exercise_social_interaction'] = X['exercise_frequency'] * X['social_interaction_score']
        
        # Health composite score
        health_cols = ['physical_health_score', 'sleep_hours', 'exercise_frequency']
        available_health_cols = [col for col in health_cols if col in X.columns]
        if len(available_health_cols) >= 2:
            X_enhanced['health_composite'] = X[available_health_cols].mean(axis=1)
        
        # Stress composite score
        stress_cols = ['work_stress_level', 'financial_stress']
        available_stress_cols = [col for col in stress_cols if col in X.columns]
        if len(available_stress_cols) >= 2:
            X_enhanced['stress_composite'] = X[available_stress_cols].mean(axis=1)
        
        # Well-being composite
        wellbeing_cols = ['relationship_satisfaction', 'social_interaction_score', 'meditation_frequency']
        available_wellbeing_cols = [col for col in wellbeing_cols if col in X.columns]
        if len(available_wellbeing_cols) >= 2:
            X_enhanced['wellbeing_composite'] = X[available_wellbeing_cols].mean(axis=1)
        
        # Age-related features
        if 'age' in X.columns:
            X_enhanced['age_group'] = pd.cut(X['age'], bins=[0, 25, 40, 60, 100], 
                                           labels=['young', 'adult', 'middle', 'senior'])
            # Convert categorical to numeric
            X_enhanced['age_group_numeric'] = pd.Categorical(X_enhanced['age_group']).codes
            X_enhanced.drop('age_group', axis=1, inplace=True)
        
        # Screen time categories
        if 'screen_time_hours' in X.columns:
            X_enhanced['excessive_screen_time'] = (X['screen_time_hours'] > 8).astype(int)
        
        # Sleep quality indicator
        if 'sleep_hours' in X.columns:
            X_enhanced['poor_sleep'] = ((X['sleep_hours'] < 6) | (X['sleep_hours'] > 9)).astype(int)
        
        logger.info(f"Enhanced features created. New shape: {X_enhanced.shape}")
        return X_enhanced
    
    def create_polynomial_features(self, X, degree=2, interaction_only=True):
        """Create polynomial features for key variables."""
        # Select key features for polynomial transformation
        key_features = ['sleep_hours', 'exercise_frequency', 'work_stress_level', 
                       'social_interaction_score', 'relationship_satisfaction']
        available_key_features = [col for col in key_features if col in X.columns]
        
        if len(available_key_features) < 2:
            logger.warning("Not enough key features for polynomial transformation")
            return X
        
        X_poly_subset = X[available_key_features]
        
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only, 
            include_bias=False
        )
        
        X_poly = self.poly_features.fit_transform(X_poly_subset)
        poly_feature_names = self.poly_features.get_feature_names_out(available_key_features)
        
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
        
        # Remove original features that are now in polynomial features
        X_remaining = X.drop(columns=available_key_features)
        
        # Combine original remaining features with polynomial features
        X_final = pd.concat([X_remaining, X_poly_df], axis=1)
        
        logger.info(f"Polynomial features created. New shape: {X_final.shape}")
        return X_final
    
    def select_features(self, X, y, k=15):
        """Select top k features using statistical tests."""
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        if hasattr(X, 'columns'):
            mask = self.feature_selector.get_support()
            self.selected_features = X.columns[mask].tolist()
            X_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        logger.info(f"Feature selection completed. Selected {X_selected.shape[1]} features")
        if self.selected_features:
            logger.info(f"Selected features: {self.selected_features}")
        
        return X_selected
    
    def create_risk_scores(self, X):
        """Create risk assessment scores based on domain knowledge."""
        X_risk = X.copy()
        
        # Mental health risk factors
        risk_factors = {}
        
        # Sleep risk
        if 'sleep_hours' in X.columns:
            risk_factors['sleep_risk'] = np.where(
                (X['sleep_hours'] < 6) | (X['sleep_hours'] > 9), 
                2, 
                np.where((X['sleep_hours'] < 7) | (X['sleep_hours'] > 8), 1, 0)
            )
        
        # Stress risk
        stress_cols = ['work_stress_level', 'financial_stress']
        available_stress = [col for col in stress_cols if col in X.columns]
        if available_stress:
            stress_mean = X[available_stress].mean(axis=1)
            risk_factors['stress_risk'] = np.where(
                stress_mean >= 8, 2,
                np.where(stress_mean >= 6, 1, 0)
            )
        
        # Social isolation risk
        if 'social_interaction_score' in X.columns:
            risk_factors['social_risk'] = np.where(
                X['social_interaction_score'] <= 3, 2,
                np.where(X['social_interaction_score'] <= 5, 1, 0)
            )
        
        # Substance use risk
        if 'substance_use_frequency' in X.columns:
            risk_factors['substance_risk'] = np.where(
                X['substance_use_frequency'] >= 4, 2,
                np.where(X['substance_use_frequency'] >= 2, 1, 0)
            )
        
        # Add risk factors to features
        for risk_name, risk_values in risk_factors.items():
            X_risk[risk_name] = risk_values
        
        # Overall risk score
        if risk_factors:
            X_risk['overall_risk_score'] = sum(risk_factors.values())
        
        logger.info(f"Risk scores created. New shape: {X_risk.shape}")
        return X_risk
    
    def engineer_all_features(self, X, y=None, create_polynomial=False):
        """Apply all feature engineering techniques."""
        logger.info("Starting comprehensive feature engineering...")
        
        # Create interaction features
        X_enhanced = self.create_interaction_features(X)
        
        # Create risk scores
        X_enhanced = self.create_risk_scores(X_enhanced)
        
        # Create polynomial features if requested
        if create_polynomial:
            X_enhanced = self.create_polynomial_features(X_enhanced)
        
        # Feature selection if target is provided
        if y is not None:
            X_enhanced = self.select_features(X_enhanced, y)
        
        logger.info(f"Feature engineering completed. Final shape: {X_enhanced.shape}")
        return X_enhanced
    
    def transform_new_data(self, X):
        """Transform new data using the fitted transformers."""
        # Apply the same transformations as during training
        X_transformed = self.create_interaction_features(X)
        X_transformed = self.create_risk_scores(X_transformed)
        
        if self.poly_features is not None:
            key_features = ['sleep_hours', 'exercise_frequency', 'work_stress_level', 
                           'social_interaction_score', 'relationship_satisfaction']
            available_key_features = [col for col in key_features if col in X_transformed.columns]
            
            if available_key_features:
                X_poly_subset = X_transformed[available_key_features]
                X_poly = self.poly_features.transform(X_poly_subset)
                poly_feature_names = self.poly_features.get_feature_names_out(available_key_features)
                
                X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
                X_remaining = X_transformed.drop(columns=available_key_features)
                X_transformed = pd.concat([X_remaining, X_poly_df], axis=1)
        
        if self.feature_selector is not None and self.selected_features is not None:
            # Ensure all selected features are present
            missing_features = set(self.selected_features) - set(X_transformed.columns)
            if missing_features:
                logger.warning(f"Missing features in new data: {missing_features}")
                # Add missing features with default values (0)
                for feature in missing_features:
                    X_transformed[feature] = 0
            
            X_transformed = X_transformed[self.selected_features]
        
        return X_transformed