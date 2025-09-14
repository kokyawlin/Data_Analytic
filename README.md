# Mental Wellness Analytics - Big Data Prediction Model

A comprehensive machine learning system for predicting mental wellness situations based on various behavioral, psychological, and lifestyle indicators. This project implements multiple ML algorithms and provides detailed insights into factors affecting mental health.

## 🎯 Project Overview

This project creates a **Big Data Analytics Model** that predicts mental wellness situations using:
- **Multiple ML Algorithms**: Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, Neural Networks
- **Advanced Feature Engineering**: Interaction features, risk scores, polynomial features
- **Comprehensive Analysis**: Data visualization, model interpretation, and personalized recommendations
- **Real-world Application**: Batch processing, individual predictions, and risk assessment

## 📊 Key Features

### Core Functionality
- **Multi-class Classification**: Predicts mental wellness as Poor, Moderate, or Good
- **Feature Engineering**: Creates meaningful interaction features and risk scores
- **Model Comparison**: Automatic selection of best-performing algorithm
- **Hyperparameter Tuning**: Optimizes model performance
- **Imbalanced Data Handling**: Uses SMOTE for balanced training

### Wellness Indicators
The model considers 12 key factors:
- Sleep patterns and quality
- Exercise frequency
- Social interaction levels
- Work and financial stress
- Screen time and digital habits
- Relationship satisfaction
- Physical health indicators
- Substance use patterns
- Meditation and mindfulness practices
- Therapy engagement

### Outputs
- **Wellness Prediction**: Poor/Moderate/Good classification
- **Confidence Scores**: Probability for each wellness category
- **Risk Assessment**: Personalized risk evaluation
- **Recommendations**: Actionable advice based on individual profile
- **Feature Importance**: Understanding of key predictors

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kokyawlin/Data_Analytic.git
cd Data_Analytic
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the main example:**
```bash
python examples/train_and_predict.py
```

### Basic Usage

```python
from mental_wellness_predictor import WellnessPredictor

# Initialize and train the model
predictor = WellnessPredictor()
results = predictor.train_model(use_feature_engineering=True)

# Make a prediction for an individual
person_data = {
    'age': 30,
    'sleep_hours': 7.0,
    'exercise_frequency': 4,
    'social_interaction_score': 7,
    'work_stress_level': 5,
    'screen_time_hours': 6.0,
    'financial_stress': 4,
    'relationship_satisfaction': 8,
    'physical_health_score': 7,
    'substance_use_frequency': 1,
    'meditation_frequency': 3,
    'therapy_sessions': 2
}

prediction = predictor.predict_single(**person_data)
print(f"Wellness: {prediction['predicted_wellness']}")
print(f"Risk: {prediction['risk_assessment']}")
print(f"Recommendations: {prediction['recommendations']}")
```

## 📁 Project Structure

```
Data_Analytic/
├── mental_wellness_predictor/     # Main package
│   ├── __init__.py               # Package initialization
│   ├── data_processor.py         # Data loading and preprocessing
│   ├── feature_engineer.py       # Feature engineering and selection
│   ├── model_trainer.py          # Model training and evaluation
│   ├── predictor.py             # Main prediction interface
│   └── visualizer.py            # Data visualization tools
├── examples/                     # Example scripts
│   ├── train_and_predict.py     # Basic usage example
│   └── batch_prediction.py      # Batch processing example
├── tests/                       # Unit tests
│   └── test_wellness_predictor.py
├── notebooks/                   # Jupyter notebooks (optional)
├── data/                       # Data directory (for custom datasets)
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔬 Model Details

### Algorithms Implemented
1. **Random Forest**: Robust ensemble method with feature importance
2. **Gradient Boosting**: Sequential learning with boosting
3. **XGBoost**: Optimized gradient boosting framework
4. **LightGBM**: Fast gradient boosting with categorical support
5. **Support Vector Machine**: Non-linear classification with RBF kernel
6. **Logistic Regression**: Linear baseline model
7. **Neural Network**: Multi-layer perceptron with early stopping

### Feature Engineering
- **Interaction Features**: Sleep-stress, exercise-social combinations
- **Composite Scores**: Health, stress, and wellbeing indices
- **Risk Indicators**: Binary flags for high-risk behaviors
- **Age Groups**: Categorical age segmentation
- **Polynomial Features**: Non-linear relationships (optional)

### Model Evaluation
- **Cross-Validation**: Stratified K-fold validation
- **Multiple Metrics**: Accuracy, precision, recall, F1-score, AUC
- **Confusion Matrix**: Detailed classification results
- **Feature Importance**: Understanding model decisions

## 📈 Visualization and Reporting

The system generates comprehensive visualizations:
- **Data Distribution**: Feature histograms and correlations
- **Model Performance**: ROC curves and confusion matrices
- **Feature Importance**: Most influential predictors
- **Prediction Confidence**: Model certainty analysis
- **Interactive Dashboard**: Plotly-based exploration tool

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/ -v
```

Or run specific tests:
```bash
python tests/test_wellness_predictor.py
```

## 📊 Example Results

### Sample Predictions

**Person 1 - Healthy Lifestyle:**
- Age: 28, Sleep: 8h, Exercise: 5x/week, Low stress
- **Prediction**: Good wellness (89% confidence)
- **Risk**: Low Risk - Continue current practices
- **Recommendations**: Maintain healthy habits, consider stress prevention

**Person 2 - High Stress Profile:**
- Age: 42, Sleep: 4h, No exercise, High work stress
- **Prediction**: Poor wellness (76% confidence)  
- **Risk**: High Risk - Immediate attention recommended
- **Recommendations**: Improve sleep hygiene, seek professional support, stress management

### Model Performance
- **Best Algorithm**: Random Forest (typical results)
- **Accuracy**: ~85-90% on test data
- **Key Features**: Sleep hours, work stress, exercise frequency
- **Cross-Validation**: Consistent performance across folds

## 🎯 Use Cases

### Healthcare Applications
- **Screening Tool**: Early identification of at-risk individuals
- **Treatment Planning**: Data-driven intervention strategies
- **Population Health**: Community wellness assessment
- **Research**: Understanding mental health determinants

### Workplace Wellness
- **Employee Assessment**: Anonymous wellness screening
- **Intervention Programs**: Targeted support initiatives
- **Policy Development**: Evidence-based workplace policies
- **ROI Analysis**: Measuring program effectiveness

### Personal Use
- **Self-Assessment**: Individual wellness monitoring
- **Lifestyle Optimization**: Data-driven habit changes
- **Progress Tracking**: Longitudinal wellness trends
- **Goal Setting**: Personalized improvement targets

## 🔧 Customization

### Using Your Own Data
```python
# Load custom dataset
predictor = WellnessPredictor()
predictor.train_model(file_path='your_data.csv', target_column='wellness')
```

### Feature Selection
```python
# Custom feature engineering
predictor.train_model(
    use_feature_engineering=True,
    tune_hyperparameters=True  # Enable for better performance
)
```

### Model Configuration
```python
# Access underlying components
predictor.model_trainer.initialize_models()
# Modify specific model parameters
predictor.model_trainer.models['random_forest'].set_params(n_estimators=200)
```

## 📋 Data Requirements

### Required Features
The model expects these 12 input features:
- `age`: Age in years (18-100)
- `sleep_hours`: Average sleep per night (0-12)
- `exercise_frequency`: Exercise sessions per week (0-7)
- `social_interaction_score`: Social engagement level (1-10)
- `work_stress_level`: Work-related stress (1-10)
- `screen_time_hours`: Daily screen time (0-16)
- `financial_stress`: Financial stress level (1-10)
- `relationship_satisfaction`: Relationship quality (1-10)
- `physical_health_score`: Overall physical health (1-10)
- `substance_use_frequency`: Substance use frequency (0-5)
- `meditation_frequency`: Meditation sessions per week (0-7)
- `therapy_sessions`: Therapy sessions per year (0-50)

### Target Variable
- `mental_wellness_category`: Poor, Moderate, Good

## ⚡ Performance Optimization

### Training Speed
- Use `tune_hyperparameters=False` for faster training
- Reduce dataset size for initial experiments
- Enable parallel processing with `n_jobs=-1`

### Memory Efficiency
- Process data in batches for large datasets
- Use feature selection to reduce dimensionality
- Consider model compression for deployment

## 🛡️ Ethical Considerations

### Privacy and Security
- **Data Anonymization**: Remove personal identifiers
- **Secure Storage**: Encrypt sensitive health data
- **Access Control**: Limit model access to authorized users
- **Audit Trail**: Log all predictions and access

### Bias and Fairness
- **Diverse Training Data**: Ensure representative samples
- **Regular Validation**: Monitor model performance across groups
- **Transparency**: Provide clear explanations of predictions
- **Human Oversight**: Always include professional judgment

### Clinical Use
⚠️ **Important**: This model is for research and screening purposes only. It should not replace professional mental health assessment or treatment decisions.

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 References

- World Health Organization Mental Health Guidelines
- American Psychological Association Assessment Standards
- Machine Learning for Healthcare Best Practices
- Ethical AI in Mental Health Applications

## 📞 Support

For questions or issues:
- Create an issue in the GitHub repository
- Check the documentation and examples
- Review the test cases for usage patterns

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Disclaimer**: This software is provided for educational and research purposes. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for mental health concerns.