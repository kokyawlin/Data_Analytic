#!/usr/bin/env python3
"""
Mental Wellness Prediction - Main Example Script

This script demonstrates how to use the Mental Wellness Predictor
to train a model and make predictions.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mental_wellness_predictor import WellnessPredictor
from mental_wellness_predictor.visualizer import WellnessVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate the mental wellness prediction system."""
    
    logger.info("Starting Mental Wellness Prediction Demo")
    
    # Initialize the predictor
    predictor = WellnessPredictor()
    
    # Train the model (this will use synthetic data since no real data file is provided)
    logger.info("Training the mental wellness prediction model...")
    
    training_results = predictor.train_model(
        use_feature_engineering=True,
        tune_hyperparameters=False  # Set to True for better results but longer training time
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model: {training_results['best_model_name']}")
    logger.info(f"Model accuracy: {training_results['evaluation_results']['accuracy']:.4f}")
    
    # Save the trained model
    model_path = "trained_wellness_model.joblib"
    predictor.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Make predictions on sample individuals
    logger.info("\n" + "="*50)
    logger.info("Making predictions on sample individuals:")
    logger.info("="*50)
    
    # Example 1: Person with good wellness indicators
    person1 = {
        'age': 28,
        'sleep_hours': 8.0,
        'exercise_frequency': 5,
        'social_interaction_score': 8,
        'work_stress_level': 3,
        'screen_time_hours': 4.0,
        'financial_stress': 2,
        'relationship_satisfaction': 9,
        'physical_health_score': 8,
        'substance_use_frequency': 0,
        'meditation_frequency': 4,
        'therapy_sessions': 0
    }
    
    prediction1 = predictor.predict_single(**person1)
    print(f"\nPerson 1 - Healthy Lifestyle:")
    print(f"Predicted Wellness: {prediction1['predicted_wellness']}")
    print(f"Risk Assessment: {prediction1['risk_assessment']}")
    print("Top Recommendations:")
    for i, rec in enumerate(prediction1['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    # Example 2: Person with moderate wellness challenges
    person2 = {
        'age': 35,
        'sleep_hours': 5.5,
        'exercise_frequency': 1,
        'social_interaction_score': 4,
        'work_stress_level': 8,
        'screen_time_hours': 10.0,
        'financial_stress': 7,
        'relationship_satisfaction': 5,
        'physical_health_score': 5,
        'substance_use_frequency': 2,
        'meditation_frequency': 0,
        'therapy_sessions': 2
    }
    
    prediction2 = predictor.predict_single(**person2)
    print(f"\nPerson 2 - Moderate Challenges:")
    print(f"Predicted Wellness: {prediction2['predicted_wellness']}")
    print(f"Risk Assessment: {prediction2['risk_assessment']}")
    print("Top Recommendations:")
    for i, rec in enumerate(prediction2['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    # Example 3: Person with concerning indicators
    person3 = {
        'age': 42,
        'sleep_hours': 4.0,
        'exercise_frequency': 0,
        'social_interaction_score': 2,
        'work_stress_level': 9,
        'screen_time_hours': 12.0,
        'financial_stress': 9,
        'relationship_satisfaction': 2,
        'physical_health_score': 3,
        'substance_use_frequency': 4,
        'meditation_frequency': 0,
        'therapy_sessions': 0
    }
    
    prediction3 = predictor.predict_single(**person3)
    print(f"\nPerson 3 - High Risk Indicators:")
    print(f"Predicted Wellness: {prediction3['predicted_wellness']}")
    print(f"Risk Assessment: {prediction3['risk_assessment']}")
    print("Top Recommendations:")
    for i, rec in enumerate(prediction3['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    # Get model insights
    logger.info("\n" + "="*50)
    logger.info("Model Insights:")
    logger.info("="*50)
    
    insights = predictor.get_model_insights()
    print(f"Model Used: {insights['model_name']}")
    print(f"Number of Features: {insights['feature_count']}")
    print(f"Target Classes: {insights['target_classes']}")
    
    if 'top_features' in insights:
        print("\nTop 5 Most Important Features:")
        for i, feature in enumerate(insights['top_features'][:5], 1):
            print(f"  {i}. {feature['feature']}: {feature['importance']:.4f}")
    
    # Create visualizations
    logger.info("\n" + "="*50)
    logger.info("Creating Visualizations:")
    logger.info("="*50)
    
    visualizer = WellnessVisualizer()
    
    # Get the training data for visualization
    data_processor = predictor.data_processor
    if hasattr(data_processor, 'data') and data_processor.data is not None:
        # Create visualization directory
        vis_dir = "./visualizations/"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate comprehensive report
        visualizer.create_comprehensive_report(
            data=data_processor.data,
            feature_importance=training_results.get('feature_importance'),
            cv_results=training_results.get('cv_results'),
            save_dir=vis_dir
        )
        
        logger.info(f"Visualizations saved to {vis_dir}")
    
    logger.info("\n" + "="*50)
    logger.info("Demo completed successfully!")
    logger.info("="*50)
    
    print(f"\nFiles created:")
    print(f"- Model: {model_path}")
    print(f"- Visualizations: ./visualizations/")
    
    print(f"\nTo use the trained model in your own code:")
    print(f"```python")
    print(f"from mental_wellness_predictor import WellnessPredictor")
    print(f"predictor = WellnessPredictor('{model_path}')")
    print(f"result = predictor.predict_single(age=30, sleep_hours=7, ...)")
    print(f"```")


if __name__ == "__main__":
    main()