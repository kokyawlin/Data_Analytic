#!/usr/bin/env python3
"""
Quick Demo Script - Mental Wellness Predictor

This script demonstrates the core functionality of the mental wellness prediction system.
"""

from mental_wellness_predictor import WellnessPredictor

def main():
    print("🧠 Mental Wellness Prediction System Demo")
    print("=" * 50)
    
    # Initialize and train the model
    predictor = WellnessPredictor()
    print("✓ Predictor initialized")
    
    print("\n📊 Training model with synthetic data...")
    results = predictor.train_model(use_feature_engineering=False, tune_hyperparameters=False)
    print(f"✓ Best model: {results['best_model_name']}")
    print(f"✓ Accuracy: {results['evaluation_results']['accuracy']:.1%}")
    
    # Demo predictions
    print("\n🔮 Making predictions for different personas:")
    print("-" * 50)
    
    # Healthy individual
    healthy_person = {
        'age': 28, 'sleep_hours': 8.0, 'exercise_frequency': 5,
        'social_interaction_score': 8, 'work_stress_level': 3,
        'screen_time_hours': 4.0, 'financial_stress': 2,
        'relationship_satisfaction': 9, 'physical_health_score': 8,
        'substance_use_frequency': 0, 'meditation_frequency': 4,
        'therapy_sessions': 0
    }
    
    pred1 = predictor.predict_single(**healthy_person)
    print(f"\n👤 Healthy Lifestyle Person:")
    print(f"   Predicted Wellness: {pred1['predicted_wellness']}")
    print(f"   Risk Level: {pred1['risk_assessment']}")
    
    # Stressed individual
    stressed_person = {
        'age': 35, 'sleep_hours': 5.0, 'exercise_frequency': 1,
        'social_interaction_score': 3, 'work_stress_level': 9,
        'screen_time_hours': 12.0, 'financial_stress': 8,
        'relationship_satisfaction': 4, 'physical_health_score': 4,
        'substance_use_frequency': 3, 'meditation_frequency': 0,
        'therapy_sessions': 0
    }
    
    pred2 = predictor.predict_single(**stressed_person)
    print(f"\n👤 High Stress Person:")
    print(f"   Predicted Wellness: {pred2['predicted_wellness']}")
    print(f"   Risk Level: {pred2['risk_assessment']}")
    print(f"   Top Recommendation: {pred2['recommendations'][0]}")
    
    print("\n✅ Demo completed successfully!")
    print("\nThe Mental Wellness Prediction System is ready for use!")
    print("Run 'python examples/train_and_predict.py' for a complete demonstration.")

if __name__ == "__main__":
    main()