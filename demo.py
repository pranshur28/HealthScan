import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from medical_classifier import MedicalDiagnosticTool

# For demo purposes, create a synthetic medical dataset
def create_synthetic_medical_dataset(n_samples=1000, n_features=20, n_informative=10, n_classes=3):
    """
    Create a synthetic dataset representing medical data for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int
        Number of patient samples
    n_features : int
        Total number of features
    n_informative : int
        Number of informative features
    n_classes : int
        Number of disease classes
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with features and target
    """
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=0,
        n_classes=n_classes,
        random_state=42,
        class_sep=1.5,
        weights=[0.6, 0.3, 0.1] if n_classes == 3 else None
    )
    
    # Create feature names
    feature_names = [
        'age', 'bmi', 'glucose', 'insulin', 'blood_pressure', 'heart_rate',
        'respiration_rate', 'temperature', 'white_blood_cell_count', 'red_blood_cell_count',
        'hemoglobin', 'platelets', 'sodium', 'potassium', 'chloride', 'calcium',
        'cholesterol', 'triglycerides', 'albumin', 'protein'
    ][:n_features]
    
    # Create disease names
    disease_names = ['Healthy', 'Type 2 Diabetes', 'Cardiovascular Disease'][:n_classes]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = [disease_names[i] for i in y]
    
    print(f"Created synthetic dataset with {n_samples} samples and {n_features} features.")
    print(f"Disease distribution:")
    print(df['diagnosis'].value_counts())
    
    return df, feature_names, 'diagnosis'

def main():
    print("Medical Diagnostic Tool Demo")
    print("===========================")
    
    # Create synthetic dataset
    print("\nStep 1: Creating synthetic medical dataset...")
    data, feature_names, target_name = create_synthetic_medical_dataset()
    
    # Initialize the diagnostic tool
    print("\nStep 2: Initializing the Medical Diagnostic Tool...")
    diagnostic_tool = MedicalDiagnosticTool()
    
    # Split features and target
    X = data[feature_names].values
    y = data[target_name].values
    
    # Train the model with hyperparameter tuning
    print("\nStep 3: Training the Random Forest classifier...")
    results = diagnostic_tool.train(X, y, feature_names, target_name, hypertune=True)
    
    print("\nClassification Report:")
    for cls, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"Class: {cls}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1-score']:.4f}")
    
    # Visualize feature importance
    print("\nStep 4: Visualizing top features...")
    fig = diagnostic_tool.visualize_feature_importance()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")
    
    # Make a prediction for a new patient
    print("\nStep 5: Making prediction for a new patient...")
    # Generate a random patient data point
    np.random.seed(42)
    new_patient = np.random.normal(0, 1, size=len(feature_names))
    prediction = diagnostic_tool.predict(new_patient)
    
    print(f"Predicted diagnosis: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("\nClass probabilities:")
    for cls, prob in prediction['top_classes'].items():
        print(f"  {cls}: {prob:.4f}")
    
    # Explain the prediction
    print("\nStep 6: Explaining the prediction...")
    explanation = diagnostic_tool.explain_prediction(new_patient)
    
    print("Top contributing features:")
    for feature, details in explanation['top_contributing_features'].items():
        print(f"  {feature}: value={details['value']:.4f}, importance={details['importance']:.4f}")
    
    # Find similar cases
    print("\nStep 7: Finding similar cases...")
    similar_cases = diagnostic_tool.find_similar_cases(new_patient, (X, y), n_neighbors=3)
    
    print("Similar cases found:")
    for i, (features, target, distance) in enumerate(zip(
            similar_cases['features'], 
            similar_cases['target'], 
            similar_cases['distances'])):
        print(f"Case {i+1}: Diagnosis: {target}, Distance: {distance:.4f}")
    
    # Save the model
    print("\nStep 8: Saving the model...")
    diagnostic_tool.save_model('medical_diagnostic_model.pkl')
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
