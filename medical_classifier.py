import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MedicalDiagnosticTool:
    """
    AI-Powered Diagnostic Tool that uses a Random Forest classifier to assist in disease diagnosis.
    Compares patient data with similar cases to support medical decision-making.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the Medical Diagnostic Tool.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to a pre-trained model file. If provided, the model will be loaded from this file.
        """
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        else:
            self.model = RandomForestClassifier(random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
            self.feature_names = None
            self.target_name = None
    
    def preprocess_data(self, X, fit=False):
        """
        Preprocess the input features by applying standard scaling.
        
        Parameters:
        -----------
        X : array-like
            Input features to preprocess.
        fit : bool, default=False
            Whether to fit the scaler on the data or just transform.
            
        Returns:
        --------
        array-like
            Preprocessed features.
        """
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def train(self, X, y, feature_names=None, target_name=None, test_size=0.2, hypertune=False):
        """
        Train the Random Forest classifier on the provided data.
        
        Parameters:
        -----------
        X : array-like
            Training features.
        y : array-like
            Target variable (disease labels).
        feature_names : list, optional
            Names of the features.
        target_name : str, optional
            Name of the target variable.
        test_size : float, default=0.2
            Proportion of data to use for testing.
        hypertune : bool, default=False
            Whether to perform hyperparameter tuning.
            
        Returns:
        --------
        dict
            Training results including accuracy and model details.
        """
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Preprocess the data
        X_train_scaled = self.preprocess_data(X_train, fit=True)
        X_test_scaled = self.preprocess_data(X_test)
        
        if hypertune:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
                scoring='accuracy'
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Model trained successfully. Test accuracy: {accuracy:.4f}")
        
        # Evaluation results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(feature_names if feature_names else [f"feature_{i}" for i in range(X.shape[1])], 
                                         self.model.feature_importances_))
        }
        
        return results
    
    def predict(self, patient_data):
        """
        Predict the diagnosis for new patient data.
        
        Parameters:
        -----------
        patient_data : array-like
            Patient's medical features.
            
        Returns:
        --------
        dict
            Prediction results including predicted disease and confidence scores.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Ensure patient_data is 2D
        if len(np.array(patient_data).shape) == 1:
            patient_data = np.array(patient_data).reshape(1, -1)
        
        # Preprocess the data
        patient_data_scaled = self.preprocess_data(patient_data)
        
        # Get prediction and probabilities
        prediction = self.model.predict(patient_data_scaled)
        probabilities = self.model.predict_proba(patient_data_scaled)
        
        # Get the class names
        class_names = self.model.classes_
        
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probabilities[0])[::-1]
        sorted_classes = class_names[sorted_indices]
        sorted_probabilities = probabilities[0][sorted_indices]
        
        # Create a dictionary of class probabilities
        class_probabilities = {class_names[i]: prob for i, prob in enumerate(probabilities[0])}
        
        result = {
            'prediction': prediction[0],
            'confidence': np.max(probabilities),
            'class_probabilities': class_probabilities,
            'top_classes': dict(zip(sorted_classes, sorted_probabilities))
        }
        
        return result
    
    def find_similar_cases(self, patient_data, dataset, n_neighbors=5):
        """
        Find similar cases from the dataset for the given patient data.
        
        Parameters:
        -----------
        patient_data : array-like
            Patient's medical features.
        dataset : pandas DataFrame
            Dataset containing features and target variables.
        n_neighbors : int, default=5
            Number of similar cases to find.
            
        Returns:
        --------
        pandas DataFrame
            Similar cases from the dataset.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Ensure patient_data is 1D
        if len(np.array(patient_data).shape) > 1:
            patient_data = np.array(patient_data).flatten()
            
        # If dataset is a pandas DataFrame, extract features
        if isinstance(dataset, pd.DataFrame):
            if self.feature_names:
                features = dataset[self.feature_names].values
                target = dataset[self.target_name].values if self.target_name else None
            else:
                # Assume all columns except the last one are features
                features = dataset.iloc[:, :-1].values
                target = dataset.iloc[:, -1].values
        else:
            # Assume dataset is a tuple of (X, y)
            features, target = dataset
        
        # Preprocess the dataset features
        features_scaled = self.scaler.transform(features)
        
        # Preprocess the patient data
        patient_data_scaled = self.scaler.transform(patient_data.reshape(1, -1))[0]
        
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((features_scaled - patient_data_scaled) ** 2, axis=1))
        
        # Get indices of the k nearest neighbors
        neighbor_indices = np.argsort(distances)[:n_neighbors]
        
        # Create DataFrame of similar cases
        if isinstance(dataset, pd.DataFrame):
            similar_cases = dataset.iloc[neighbor_indices].copy()
            similar_cases['distance'] = distances[neighbor_indices]
            return similar_cases
        else:
            similar_features = features[neighbor_indices]
            similar_target = target[neighbor_indices] if target is not None else None
            return {
                'features': similar_features,
                'target': similar_target,
                'distances': distances[neighbor_indices]
            }
    
    def explain_prediction(self, patient_data):
        """
        Provide an explanation for the model's prediction based on feature importance.
        
        Parameters:
        -----------
        patient_data : array-like
            Patient's medical features.
            
        Returns:
        --------
        dict
            Explanation of the prediction including important features.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Ensure patient_data is 2D
        if len(np.array(patient_data).shape) == 1:
            patient_data = np.array(patient_data).reshape(1, -1)
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        # Get feature names or create generic ones
        feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Calculate contribution of each feature to the prediction
        patient_data_scaled = self.preprocess_data(patient_data)[0]
        
        # Create a dictionary of feature values and their importance
        feature_contributions = {}
        for i, feature in enumerate(feature_names):
            contribution = patient_data_scaled[i] * feature_importance[i]
            feature_contributions[feature] = {
                'value': patient_data[0][i],
                'importance': feature_importance[i],
                'contribution': contribution
            }
        
        # Sort features by contribution
        sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]['contribution']), reverse=True)
        top_features = dict(sorted_features[:5])
        
        explanation = {
            'top_contributing_features': top_features,
            'feature_importance': dict(zip(feature_names, feature_importance))
        }
        
        return explanation
    
    def save_model(self, model_path):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Cannot save untrained model.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, model_path.replace('.pkl', '_scaler.pkl'))
        
        print(f"Model saved to {model_path}")
    
    def visualize_feature_importance(self, top_n=10):
        """
        Visualize the importance of top N features.
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top features to visualize.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the feature importance plot.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        # Get feature names or create generic ones
        feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Create a DataFrame of feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Visualize the confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        class_names : list, optional
            Names of the classes.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the confusion matrix plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names if class_names else "auto",
                   yticklabels=class_names if class_names else "auto")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        return plt.gcf()
