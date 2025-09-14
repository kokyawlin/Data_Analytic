"""
Wellness Predictor Module

This module provides the main interface for making mental wellness predictions.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Union, Optional
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class WellnessPredictor:
    """Main predictor class for mental wellness assessment."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        self.is_trained = False
        self.feature_names = None
        self.target_classes = None
        
        if model_path:
            self.load_model(model_path)
    
    def train_model(self, data=None, file_path=None, target_column='mental_wellness_category',
                   test_size=0.2, use_feature_engineering=True, tune_hyperparameters=False):
        """Train the mental wellness prediction model."""
        logger.info("Starting model training pipeline...")
        
        # Load and process data
        self.data_processor.load_data(file_path=file_path, data=data)
        cleaned_data = self.data_processor.clean_data()
        X, y = self.data_processor.prepare_features_target(target_column)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Feature engineering
        if use_feature_engineering:
            logger.info("Applying feature engineering...")
            X_train = self.feature_engineer.engineer_all_features(X_train, y_train)
            X_test = self.feature_engineer.transform_new_data(X_test)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.data_processor.scale_features(X_train, X_test)
        
        # Store feature names
        self.feature_names = X_train_scaled.columns.tolist()
        self.target_classes = self.data_processor.get_target_classes()
        
        # Train models
        self.model_trainer.initialize_models()
        self.model_trainer.cross_validate_models(X_train_scaled, y_train)
        
        # Hyperparameter tuning if requested
        if tune_hyperparameters:
            self.model_trainer.hyperparameter_tuning(X_train_scaled, y_train)
        
        # Train best model
        self.model_trainer.train_best_model(X_train_scaled, y_train)
        
        # Evaluate model
        evaluation_results = self.model_trainer.evaluate_model(
            self.model_trainer.best_model, X_test_scaled, y_test,
            f"Best Model ({self.model_trainer.best_model_name})"
        )
        
        # Get feature importance
        feature_importance = self.model_trainer.get_feature_importance(self.feature_names)
        
        self.is_trained = True
        logger.info("Model training completed successfully!")
        
        return {
            'evaluation_results': evaluation_results,
            'feature_importance': feature_importance,
            'best_model_name': self.model_trainer.best_model_name,
            'cv_results': self.model_trainer.cv_results
        }
    
    def predict(self, input_data: Union[Dict, pd.DataFrame, List[Dict]]) -> Dict:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()
        
        # Process data
        original_data = df.copy()
        
        # Apply feature engineering
        df_engineered = self.feature_engineer.transform_new_data(df)
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df_engineered.columns)
            if missing_features:
                logger.warning(f"Missing features, filling with zeros: {missing_features}")
                for feature in missing_features:
                    df_engineered[feature] = 0
            
            # Select only the features used during training
            df_engineered = df_engineered[self.feature_names]
        
        # Scale features
        df_scaled = self.data_processor.scale_features(df_engineered)
        
        # Make predictions
        predictions = self.model_trainer.best_model.predict(df_scaled)
        prediction_probabilities = self.model_trainer.best_model.predict_proba(df_scaled)
        
        # Convert predictions back to original labels
        if self.target_classes is not None:
            predicted_labels = [self.target_classes[pred] for pred in predictions]
        else:
            predicted_labels = predictions.tolist()
        
        # Create results
        results = []
        for i in range(len(df)):
            result = {
                'predicted_wellness': predicted_labels[i],
                'confidence_scores': {
                    class_name: float(prob) for class_name, prob 
                    in zip(self.target_classes if self.target_classes is not None else ['Class_0', 'Class_1', 'Class_2'], 
                          prediction_probabilities[i])
                },
                'risk_assessment': self._assess_risk(predicted_labels[i], prediction_probabilities[i]),
                'recommendations': self._get_recommendations(original_data.iloc[i], predicted_labels[i])
            }
            results.append(result)
        
        return {
            'predictions': results,
            'model_used': self.model_trainer.best_model_name,
            'total_samples': len(df)
        }
    
    def predict_single(self, **kwargs) -> Dict:
        """Make a prediction for a single individual."""
        return self.predict(kwargs)['predictions'][0]
    
    def _assess_risk(self, prediction, probabilities):
        """Assess risk level based on prediction and confidence."""
        max_prob = max(probabilities)
        
        if prediction == 'Poor' or (hasattr(self.target_classes, '__getitem__') and 
                                   self.target_classes is not None and 
                                   prediction == self.target_classes[0]):
            if max_prob > 0.8:
                return "High Risk - Immediate attention recommended"
            elif max_prob > 0.6:
                return "Moderate Risk - Professional consultation advised"
            else:
                return "Uncertain - Further assessment needed"
        elif prediction == 'Moderate' or (hasattr(self.target_classes, '__getitem__') and 
                                         self.target_classes is not None and 
                                         len(self.target_classes) > 1 and 
                                         prediction == self.target_classes[1]):
            if max_prob > 0.7:
                return "Moderate Risk - Lifestyle improvements recommended"
            else:
                return "Low-Moderate Risk - Monitor wellness indicators"
        else:  # Good wellness
            if max_prob > 0.7:
                return "Low Risk - Continue current wellness practices"
            else:
                return "Uncertain - Consider wellness check-in"
    
    def _get_recommendations(self, individual_data, prediction):
        """Generate personalized recommendations based on prediction and individual data."""
        recommendations = []
        
        # General recommendations based on prediction
        if prediction == 'Poor' or (hasattr(self.target_classes, '__getitem__') and 
                                   self.target_classes is not None and 
                                   prediction == self.target_classes[0]):
            recommendations.extend([
                "Consider seeking professional mental health support",
                "Prioritize stress management techniques",
                "Establish a consistent daily routine"
            ])
        
        # Specific recommendations based on individual factors
        if 'sleep_hours' in individual_data:
            sleep_hours = individual_data['sleep_hours']
            if sleep_hours < 6:
                recommendations.append("Improve sleep hygiene - aim for 7-9 hours per night")
            elif sleep_hours > 9:
                recommendations.append("Consider evaluating sleep quality and potential sleep disorders")
        
        if 'exercise_frequency' in individual_data:
            exercise = individual_data['exercise_frequency']
            if exercise < 3:
                recommendations.append("Increase physical activity - aim for at least 3-4 sessions per week")
        
        if 'work_stress_level' in individual_data:
            stress = individual_data['work_stress_level']
            if stress >= 7:
                recommendations.append("Implement workplace stress management strategies")
        
        if 'social_interaction_score' in individual_data:
            social = individual_data['social_interaction_score']
            if social <= 4:
                recommendations.append("Increase social connections and community engagement")
        
        if 'substance_use_frequency' in individual_data:
            substance = individual_data['substance_use_frequency']
            if substance >= 3:
                recommendations.append("Consider reducing substance use and seeking support if needed")
        
        if not recommendations:
            recommendations.append("Continue maintaining healthy lifestyle habits")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_model_insights(self):
        """Get insights about the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        insights = {
            'model_name': self.model_trainer.best_model_name,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'target_classes': self.target_classes,
            'cv_results': self.model_trainer.cv_results
        }
        
        # Add feature importance if available
        if self.feature_names:
            feature_importance = self.model_trainer.get_feature_importance(self.feature_names)
            if feature_importance is not None:
                insights['top_features'] = feature_importance.head(10).to_dict('records')
        
        return insights
    
    def save_model(self, filepath: str):
        """Save the complete trained model pipeline."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model_trainer': self.model_trainer,
            'data_processor': self.data_processor,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'is_trained': self.is_trained
        }
        
        try:
            joblib.dump(model_data, filepath)
            logger.info(f"Complete model pipeline saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model pipeline: {e}")
            return False
    
    def load_model(self, filepath: str):
        """Load a complete trained model pipeline."""
        try:
            model_data = joblib.load(filepath)
            
            self.model_trainer = model_data['model_trainer']
            self.data_processor = model_data['data_processor']
            self.feature_engineer = model_data['feature_engineer']
            self.feature_names = model_data['feature_names']
            self.target_classes = model_data['target_classes']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Complete model pipeline loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model pipeline: {e}")
            return False