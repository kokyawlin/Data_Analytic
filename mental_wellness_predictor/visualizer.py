"""
Visualization Module for Mental Wellness Analysis

This module provides various visualization tools for data exploration and model insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class WellnessVisualizer:
    """Provides visualization tools for mental wellness data and model results."""
    
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    def plot_data_distribution(self, data, save_path=None):
        """Plot distribution of key features."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        n_cols = 3
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                axes[i].hist(data[col], bins=30, color=self.color_palette[i % len(self.color_palette)], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, data, save_path=None):
        """Plot correlation matrix of features."""
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.1, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_target_distribution(self, y, class_names=None, save_path=None):
        """Plot distribution of target variable."""
        if isinstance(y, pd.Series):
            y = y.values
        
        unique_values, counts = np.unique(y, return_counts=True)
        
        if class_names and len(class_names) == len(unique_values):
            labels = class_names
        else:
            labels = [f'Class {val}' for val in unique_values]
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(labels, counts, color=self.color_palette[:len(unique_values)])
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.title('Distribution of Mental Wellness Categories')
        plt.xlabel('Wellness Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df, top_n=15, save_path=None):
        """Plot feature importance."""
        if feature_importance_df is None:
            logger.warning("No feature importance data provided")
            return
        
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.color_palette[0])
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_pred_proba, class_names=None, save_path=None):
        """Plot ROC curves for multiclass classification."""
        if y_pred_proba is None:
            logger.warning("No probability predictions provided for ROC curves")
            return
        
        n_classes = y_pred_proba.shape[1]
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                    color=self.color_palette[i % len(self.color_palette)],
                    label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Mental Wellness Classification')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_wellness_factors(self, data, target_col='mental_wellness_category', save_path=None):
        """Plot wellness factors by category."""
        key_factors = ['sleep_hours', 'exercise_frequency', 'work_stress_level', 
                      'social_interaction_score', 'relationship_satisfaction']
        
        available_factors = [col for col in key_factors if col in data.columns]
        
        if not available_factors:
            logger.warning("No key wellness factors found in data")
            return
        
        n_factors = len(available_factors)
        fig, axes = plt.subplots(2, (n_factors + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_factors > 1 else [axes]
        
        for i, factor in enumerate(available_factors):
            if i < len(axes):
                sns.boxplot(data=data, x=target_col, y=factor, ax=axes[i])
                axes[i].set_title(f'{factor} by Wellness Category')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(available_factors), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, data, feature_importance=None, save_path=None):
        """Create an interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Target Distribution', 'Feature Correlation', 
                          'Key Wellness Factors', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # Target distribution
        if 'mental_wellness_category' in data.columns:
            target_counts = data['mental_wellness_category'].value_counts()
            fig.add_trace(
                go.Bar(x=target_counts.index, y=target_counts.values, 
                      name="Target Distribution",
                      marker_color=self.color_palette[0]),
                row=1, col=1
            )
        
        # Correlation heatmap
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.index,
                          colorscale='RdBu',
                          name="Correlation"),
                row=1, col=2
            )
        
        # Box plots for key factors
        if 'sleep_hours' in data.columns and 'mental_wellness_category' in data.columns:
            for category in data['mental_wellness_category'].unique():
                subset = data[data['mental_wellness_category'] == category]
                fig.add_trace(
                    go.Box(y=subset['sleep_hours'], name=f'{category}',
                          marker_color=self.color_palette[hash(category) % len(self.color_palette)]),
                    row=2, col=1
                )
        
        # Feature importance
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            fig.add_trace(
                go.Bar(x=top_features['importance'], y=top_features['feature'],
                      orientation='h', name="Feature Importance",
                      marker_color=self.color_palette[2]),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Mental Wellness Analytics Dashboard",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig
    
    def plot_prediction_confidence(self, predictions_data, save_path=None):
        """Plot prediction confidence distribution."""
        if not predictions_data or 'predictions' not in predictions_data:
            logger.warning("No prediction data provided")
            return
        
        confidence_scores = []
        predicted_classes = []
        
        for pred in predictions_data['predictions']:
            max_confidence = max(pred['confidence_scores'].values())
            predicted_class = pred['predicted_wellness']
            confidence_scores.append(max_confidence)
            predicted_classes.append(predicted_class)
        
        # Create confidence distribution plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidence_scores, bins=20, color=self.color_palette[0], alpha=0.7)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        
        plt.subplot(1, 2, 2)
        df_pred = pd.DataFrame({
            'predicted_class': predicted_classes,
            'confidence': confidence_scores
        })
        
        for cls in df_pred['predicted_class'].unique():
            subset = df_pred[df_pred['predicted_class'] == cls]
            plt.hist(subset['confidence'], alpha=0.7, 
                    label=f'{cls}', bins=15)
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence by Predicted Class')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, cv_results, save_path=None):
        """Plot comparison of different models."""
        if not cv_results:
            logger.warning("No cross-validation results provided")
            return
        
        models = list(cv_results.keys())
        mean_scores = [results['mean_cv_score'] for results in cv_results.values()]
        std_scores = [results['std_cv_score'] for results in cv_results.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, mean_scores, yerr=std_scores, capsize=5,
                      color=self.color_palette[:len(models)], alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, mean_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_comprehensive_report(self, data, predictions_data=None, 
                                  feature_importance=None, cv_results=None, 
                                  save_dir='./visualizations/'):
        """Create a comprehensive visualization report."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info("Generating comprehensive visualization report...")
        
        # Data distribution
        self.plot_data_distribution(data, save_path=f'{save_dir}data_distribution.png')
        
        # Correlation matrix
        self.plot_correlation_matrix(data, save_path=f'{save_dir}correlation_matrix.png')
        
        # Target distribution
        if 'mental_wellness_category' in data.columns:
            self.plot_target_distribution(data['mental_wellness_category'], 
                                        save_path=f'{save_dir}target_distribution.png')
        
        # Wellness factors
        self.plot_wellness_factors(data, save_path=f'{save_dir}wellness_factors.png')
        
        # Feature importance
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance, 
                                       save_path=f'{save_dir}feature_importance.png')
        
        # Model comparison
        if cv_results:
            self.plot_model_comparison(cv_results, 
                                     save_path=f'{save_dir}model_comparison.png')
        
        # Prediction confidence
        if predictions_data:
            self.plot_prediction_confidence(predictions_data, 
                                          save_path=f'{save_dir}prediction_confidence.png')
        
        # Interactive dashboard
        self.create_interactive_dashboard(data, feature_importance, 
                                        save_path=f'{save_dir}interactive_dashboard.html')
        
        logger.info(f"Comprehensive report saved to {save_dir}")