#!/usr/bin/env python3
"""
Batch Prediction Example

This script demonstrates how to make predictions on a batch of individuals
and generate a report.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mental_wellness_predictor import WellnessPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_batch_data():
    """Create sample data for batch prediction."""
    
    sample_individuals = [
        # Young professional with good habits
        {
            'age': 26,
            'sleep_hours': 7.5,
            'exercise_frequency': 4,
            'social_interaction_score': 8,
            'work_stress_level': 4,
            'screen_time_hours': 5,
            'financial_stress': 3,
            'relationship_satisfaction': 8,
            'physical_health_score': 8,
            'substance_use_frequency': 1,
            'meditation_frequency': 3,
            'therapy_sessions': 0,
            'name': 'Alex (Young Professional)'
        },
        
        # Middle-aged parent with stress
        {
            'age': 38,
            'sleep_hours': 6,
            'exercise_frequency': 2,
            'social_interaction_score': 5,
            'work_stress_level': 7,
            'screen_time_hours': 8,
            'financial_stress': 6,
            'relationship_satisfaction': 6,
            'physical_health_score': 6,
            'substance_use_frequency': 2,
            'meditation_frequency': 1,
            'therapy_sessions': 3,
            'name': 'Jordan (Parent)'
        },
        
        # Senior with health focus
        {
            'age': 62,
            'sleep_hours': 8,
            'exercise_frequency': 5,
            'social_interaction_score': 7,
            'work_stress_level': 2,
            'screen_time_hours': 3,
            'financial_stress': 2,
            'relationship_satisfaction': 9,
            'physical_health_score': 7,
            'substance_use_frequency': 0,
            'meditation_frequency': 6,
            'therapy_sessions': 1,
            'name': 'Pat (Senior)'
        },
        
        # College student with challenges
        {
            'age': 20,
            'sleep_hours': 5,
            'exercise_frequency': 1,
            'social_interaction_score': 4,
            'work_stress_level': 8,
            'screen_time_hours': 12,
            'financial_stress': 8,
            'relationship_satisfaction': 3,
            'physical_health_score': 4,
            'substance_use_frequency': 3,
            'meditation_frequency': 0,
            'therapy_sessions': 0,
            'name': 'Casey (Student)'
        },
        
        # Remote worker with isolation
        {
            'age': 32,
            'sleep_hours': 6.5,
            'exercise_frequency': 1,
            'social_interaction_score': 2,
            'work_stress_level': 6,
            'screen_time_hours': 11,
            'financial_stress': 5,
            'relationship_satisfaction': 4,
            'physical_health_score': 5,
            'substance_use_frequency': 2,
            'meditation_frequency': 0,
            'therapy_sessions': 2,
            'name': 'Taylor (Remote Worker)'
        }
    ]
    
    return pd.DataFrame(sample_individuals)


def main():
    """Main function for batch prediction demo."""
    
    logger.info("Starting Batch Prediction Demo")
    
    # Initialize and train predictor
    predictor = WellnessPredictor()
    
    logger.info("Training model...")
    training_results = predictor.train_model(
        use_feature_engineering=True,
        tune_hyperparameters=False
    )
    
    # Create sample batch data
    batch_data = create_sample_batch_data()
    logger.info(f"Created batch data for {len(batch_data)} individuals")
    
    # Extract names and remove from prediction data
    names = batch_data['name'].tolist()
    prediction_data = batch_data.drop('name', axis=1)
    
    # Make batch predictions
    logger.info("Making batch predictions...")
    results = predictor.predict(prediction_data)
    
    # Generate report
    logger.info("\n" + "="*70)
    logger.info("MENTAL WELLNESS BATCH PREDICTION REPORT")
    logger.info("="*70)
    
    for i, (name, prediction) in enumerate(zip(names, results['predictions'])):
        print(f"\n{i+1}. {name}")
        print("-" * (len(name) + 4))
        print(f"Predicted Wellness: {prediction['predicted_wellness']}")
        print(f"Risk Assessment: {prediction['risk_assessment']}")
        
        print("Confidence Scores:")
        for class_name, confidence in prediction['confidence_scores'].items():
            print(f"  - {class_name}: {confidence:.1%}")
        
        print("Recommendations:")
        for j, rec in enumerate(prediction['recommendations'][:3], 1):
            print(f"  {j}. {rec}")
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("BATCH SUMMARY")
    logger.info("="*70)
    
    wellness_counts = {}
    risk_levels = {}
    
    for prediction in results['predictions']:
        wellness = prediction['predicted_wellness']
        risk = prediction['risk_assessment'].split(' -')[0]  # Get risk level only
        
        wellness_counts[wellness] = wellness_counts.get(wellness, 0) + 1
        risk_levels[risk] = risk_levels.get(risk, 0) + 1
    
    print(f"\nWellness Distribution:")
    for wellness, count in wellness_counts.items():
        percentage = (count / len(results['predictions'])) * 100
        print(f"  {wellness}: {count} individuals ({percentage:.1f}%)")
    
    print(f"\nRisk Level Distribution:")
    for risk, count in risk_levels.items():
        percentage = (count / len(results['predictions'])) * 100
        print(f"  {risk}: {count} individuals ({percentage:.1f}%)")
    
    # Save detailed results to CSV
    detailed_results = []
    for i, (name, prediction) in enumerate(zip(names, results['predictions'])):
        row = {
            'Name': name,
            'Age': batch_data.iloc[i]['age'],
            'Predicted_Wellness': prediction['predicted_wellness'],
            'Risk_Assessment': prediction['risk_assessment'],
            'Max_Confidence': max(prediction['confidence_scores'].values()),
            'Top_Recommendation': prediction['recommendations'][0] if prediction['recommendations'] else ''
        }
        # Add confidence scores
        for class_name, confidence in prediction['confidence_scores'].items():
            row[f'Confidence_{class_name}'] = confidence
        
        detailed_results.append(row)
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('batch_prediction_results.csv', index=False)
    
    logger.info(f"\nDetailed results saved to: batch_prediction_results.csv")
    logger.info(f"Model used: {results['model_used']}")
    logger.info(f"Total individuals processed: {results['total_samples']}")
    
    # Create visualization of results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Wellness distribution pie chart
        wellness_labels = list(wellness_counts.keys())
        wellness_values = list(wellness_counts.values())
        axes[0].pie(wellness_values, labels=wellness_labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Wellness Distribution')
        
        # Risk levels bar chart
        risk_labels = list(risk_levels.keys())
        risk_values = list(risk_levels.values())
        axes[1].bar(risk_labels, risk_values, color=['red', 'orange', 'yellow', 'green'])
        axes[1].set_title('Risk Level Distribution')
        axes[1].set_ylabel('Number of Individuals')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('batch_prediction_summary.png', dpi=300, bbox_inches='tight')
        logger.info("Summary visualization saved to: batch_prediction_summary.png")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
    
    logger.info("\nBatch prediction demo completed successfully!")


if __name__ == "__main__":
    main()