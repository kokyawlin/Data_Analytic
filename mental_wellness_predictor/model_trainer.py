"""
Model Training Module for Mental Wellness Prediction

This module handles model training, validation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation for mental wellness prediction."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.smote = SMOTE(random_state=42)
        self.cv_results = {}
    
    def initialize_models(self):
        """Initialize different ML models with default parameters."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def handle_imbalanced_data(self, X, y):
        """Handle imbalanced data using SMOTE."""
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Original class distribution: {class_distribution}")
        
        # Check if we have enough samples for SMOTE
        min_samples = min(counts)
        if min_samples < 6:  # SMOTE needs at least 6 samples
            logger.warning(f"Not enough samples for SMOTE (min class has {min_samples} samples). Skipping resampling.")
            return X, y
        
        # Apply SMOTE if classes are imbalanced
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 1.5:
            logger.info("Applying SMOTE to handle imbalanced data")
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            
            # Ensure y_resampled is numeric (SMOTE might convert it)
            if hasattr(y_resampled, 'dtype') and y_resampled.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_resampled = le.fit_transform(y_resampled)
            
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            new_distribution = dict(zip(unique_new, counts_new))
            logger.info(f"New class distribution: {new_distribution}")
            
            return X_resampled, y_resampled
        
        return X, y
    
    def train_single_model(self, model_name: str, X_train, y_train, X_val=None, y_val=None):
        """Train a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_score = model.score(X_train, y_train)
        logger.info(f"{model_name} training accuracy: {train_score:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"{model_name} validation accuracy: {val_score:.4f}")
            return model, train_score, val_score
        
        return model, train_score
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Perform cross-validation on all models."""
        if not self.models:
            self.initialize_models()
        
        # Handle imbalanced data
        X_balanced, y_balanced = self.handle_imbalanced_data(X, y)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            logger.info(f"Cross-validating {model_name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='accuracy')
            
            self.cv_results[model_name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            logger.info(f"{model_name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model based on CV results
        best_cv_score = max(self.cv_results.values(), key=lambda x: x['mean_cv_score'])['mean_cv_score']
        self.best_model_name = next(
            name for name, results in self.cv_results.items() 
            if results['mean_cv_score'] == best_cv_score
        )
        
        logger.info(f"Best model: {self.best_model_name} with CV score: {best_cv_score:.4f}")
    
    def hyperparameter_tuning(self, X, y, model_name=None):
        """Perform hyperparameter tuning for specified model or best model."""
        if model_name is None:
            model_name = self.best_model_name or 'random_forest'
        
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Handle imbalanced data
        X_balanced, y_balanced = self.handle_imbalanced_data(X, y)
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        # Perform grid search
        base_model = self.models[model_name]
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_balanced, y_balanced)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def train_best_model(self, X_train, y_train, use_balancing=True):
        """Train the best model with optimal parameters."""
        if self.best_model_name is None:
            self.best_model_name = 'random_forest'
            logger.warning("No best model selected, using Random Forest as default")
        
        # Handle imbalanced data if requested
        if use_balancing:
            X_train, y_train = self.handle_imbalanced_data(X_train, y_train)
        
        # Train the best model
        self.best_model = self.models[self.best_model_name]
        self.best_model.fit(X_train, y_train)
        
        logger.info(f"Best model ({self.best_model_name}) trained successfully")
        return self.best_model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance on test data."""
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate AUC for multiclass
        auc = None
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except ValueError:
                logger.warning("Could not calculate AUC score")
        
        # Print results
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        if auc:
            logger.info(f"AUC: {auc:.4f}")
        
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model."""
        if self.best_model is None:
            logger.error("No trained model available")
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 most important features:")
            logger.info(feature_importance.head(10))
            
            return feature_importance
        else:
            logger.warning("Best model does not support feature importance")
            return None
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.best_model is None:
            logger.error("No trained model to save")
            return False
        
        try:
            joblib.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'cv_results': self.cv_results
            }, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.cv_results = model_data.get('cv_results', {})
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False