"""
Genre Classification Module
===========================

This module implements various machine learning models for music genre 
classification using the GTZAN dataset.

Models included:
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- Multi-Layer Perceptron (MLP)
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class GenreClassifier:
    """
    Class used to create and evaluate genre classification models.

    This class provides methods to train multiple classifiers, compare their
    performance, visualize results, and persist the best model.

    Attributes
    ----------
    model_directory : Path
        Directory to save trained models
    models : dict of str to sklearn estimator
        Dictionary of ML models to train
    best_model : sklearn estimator or None
        The best performing model after training
    best_model_name : str or None
        Name of the best model
    label_encoder : LabelEncoder or None
        Encoder for genre labels
    results : dict
        Training results for all models
    """

    def __init__(self, model_directory: str = '/files/project-MOISE/results/models') -> None:
        """
        Initialize the GenreClassifier.

        Parameters
        ----------
        model_directory : str, default='/files/project-MOISE/results/models'
            Directory to save trained models. Created if it doesn't exist.
        """
        self.model_directory: Path = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.results: Dict[str, Any] = {}
        print(f"Model directory set to: {self.model_directory}")
        
    def create_models(self) -> Dict[str, Any]:
        """
        Create a dictionary of machine learning models for genre classification.

        Instantiates four classifiers with tuned hyperparameters based on
        typical best practices for audio classification tasks.

        Returns
        -------
        dict of str to sklearn estimator
            Dictionary mapping model names to model instances:
            - 'RandomForest': RandomForestClassifier with n_jobs=-1
            - 'SVM': SVC with RBF kernel
            - 'GradientBoosting': GradientBoostingClassifier
            - 'MLP': MLPClassifier with early stopping

        Notes
        -----
        RandomForest uses n_jobs=-1 for parallel tree construction.
        """
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
                ),
            'MLP': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42, early_stopping=True)
        }
        print("Models created: ", list(self.models.keys())," for comparison.")
        return self.models
    
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        label_encoder: Optional[LabelEncoder] = None
        ) -> Dict[str, Any]:
        """
        Train all models and evaluate their performance.

        For each model, computes training/test accuracy, precision, recall,
        F1-score, and 5-fold cross-validation scores. Selects the best model
        based on the average of test accuracy and CV mean accuracy.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : np.ndarray
            Training labels (encoded as integers)
        X_test : pd.DataFrame
            Test features
        y_test : np.ndarray
            Test labels (encoded as integers)
        label_encoder : LabelEncoder, optional
            Fitted label encoder for decoding predictions back to class names

        Returns
        -------
        dict of str to dict
            Dictionary mapping model names to result dictionaries containing:
            - 'model': trained model instance
            - 'train_accuracy': float
            - 'test_accuracy': float
            - 'precision': float (weighted average)
            - 'recall': float (weighted average)
            - 'f1': float (weighted average)
            - 'cv_mean': float (mean CV accuracy)
            - 'cv_std': float (CV standard deviation)
            - 'predictions': np.ndarray (test set predictions)

        Raises
        ------
        ValueError
            If input data is invalid (wrong type, empty, or mismatched lengths)
        RuntimeError
            If no models are successfully trained
        """
        # Validate inputs
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a pandas DataFrame")
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Training or test data is empty")
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
        if len(X_test) != len(y_test):
            raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")

        if not self.models:
            self.create_models()
        
        self.label_encoder = label_encoder
        results = {}
        best_score = 0

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)

                # Calculate precision, recall, F1 (weighted average across classes)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted', zero_division=0)
                
                # Cross-validation (on training set)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
                
                results[model_name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred_test
                    }
                
                # Track best model
                combined_score = (test_accuracy + cv_scores.mean()) / 2
                if combined_score > best_score:
                        best_score = combined_score
                        self.best_model = model
                        self.best_model_name = model_name

                # Print results (.2% used for formatting)
                print(f"Training Accuracy: {train_accuracy:.2%}")
                print(f"Test Accuracy: {test_accuracy:.2%}")
                print(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1-Score: {f1:.2%}")
                print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                print("Skipping to next model.")
                continue
        
        if not results:
            raise RuntimeError("No models were successfully trained.")

        print("\n")
        print(f"BEST MODEL: {self.best_model_name} (Combined score: {best_score:.4f})")
        print("=" * 70)

        self.results = results
        return results

    def plot_model_comparison(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path]=None,
        show: bool=False
        ) -> plt.Figure:
        """
        Plot comparison of model performances.

        Creates a two-panel figure showing:
        1. Train vs test accuracy bar chart
        2. Cross-validation scores with error bars

        Parameters
        ----------
        results : dict, optional
            Results dictionary from train_and_evaluate. If None, uses
            self.results from the last training run.
        save_path : Path or str, optional
            Path to save the figure. If None, saves to
            '{model_directory}/../figures/04_model_comparison.png'
        show : bool, default=False
            Whether to display the plot interactively

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object

        Raises
        ------
        ValueError
            If no results are available to plot
        """
        if results is None:
            results = self.results
        if not results:
            raise ValueError("No results to plot. Please run train_and_evaluate first.")
        if save_path is None:
            save_path = self.model_directory.parent/ 'figures' / '04_model_comparison.png'
        else:
            save_path = Path(save_path)
        
        #Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        model_names = list(results.keys())
        train_accs = [results[m]['train_accuracy'] for m in model_names]
        test_accs = [results[m]['test_accuracy'] for m in model_names]
        cv_mean = [results[m]['cv_mean'] for m in model_names]
        cv_stds = [results[m]['cv_std'] for m in model_names]
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy comparison
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0].bar(x - width/2, train_accs, width, label='Train Accuracy', color='skyblue')        
        axes[0].bar(x + width/2, test_accs, width, label='Test Accuracy', color='coral')
        axes[0].set_xlabel('Model', fontsize = 14)
        axes[0].set_ylabel('Accuracy', fontsize = 14)
        axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')

        # Subplot 2: Cross-Validation Scores with error bars
        axes[1].bar(x, cv_mean, color='mediumseagreen', alpha=0.7)
        axes[1].errorbar(x, cv_mean, yerr=cv_stds, fmt='none', ecolor='black', capsize=5, capthick=2)
        axes[1].set_xlabel('Model', fontsize=14)
        axes[1].set_ylabel('CV Accuracy', fontsize=14)
        axes[1].set_title('Cross-Validation Performance', fontweight='bold', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].grid(alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()
        print(f"Model comparison plot saved to {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path]=None,
        show: bool=False
        ) -> None:
        """
        Plot confusion matrix for the best model.

        Creates a heatmap visualization of the confusion matrix with
        class labels on both axes.

        Parameters
        ----------
        y_test : np.ndarray
            True labels (encoded as integers)
        y_pred : np.ndarray
            Predicted labels from the model
        save_path : Path or str, optional
            Path to save the figure. If None, saves to
            '{model_directory}/../figures/05_confusion_matrix.png'
        show : bool, default=False
            Whether to display the plot interactively
        """
        if save_path is None:
            save_path = self.model_directory.parent / 'figures' / '05_confusion_matrix.png'
        else:
            save_path = Path(save_path)
        
        #Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_test, y_pred)
        
        # Get class names if label encoder available
        if self.label_encoder:
            class_names = self.label_encoder.classes_
        else:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_test)))]
        
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names,cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontweight='bold', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Confusion matrix saved to {save_path}")
        return fig
    
    def save_model(self, filename: Optional[str]=None) -> None:
        """
        Save the best trained model using joblib serialization.

        Saves the model along with metadata including model name,
        label encoder, and training results for reproducibility.

        Parameters
        ----------
        filename : str, optional
            Custom filename (without path). If None, uses
            'best_genre_classifier_{model_name}.pkl'

        Raises
        ------
        RuntimeError
            If no model has been trained yet

        Notes
        -----
        The saved file contains a dictionary with keys:
        - 'model': the trained sklearn estimator
        - 'model_name': string identifier
        - 'label_encoder': fitted LabelEncoder
        - 'results': full training results dictionary
        """
        if self.best_model is None:
            raise RuntimeError("No model has been trained yet. Train a model first.")
        
        if filename is None:
            filename = f"best_genre_classifier_{self.best_model_name.replace(' ', '_').lower()}.pkl"
        
        filepath = self.model_directory / filename
        
        # Save model, metadata, and encoder
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'label_encoder': self.label_encoder,
            'results': self.results
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model.

        Restores the model, model name, label encoder, and results
        from a joblib-serialized file.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model file (.pkl)

        Raises
        ------
        FileNotFoundError
            If the model file does not exist
        RuntimeError
            If loading fails due to file corruption or incompatibility
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            data = joblib.load(filepath)
            self.best_model = data['model']
            self.best_model_name = data['model_name']
            self.label_encoder = data.get('label_encoder')
            self.results = data.get('results', {})
            print(f"Model loaded from {filepath}")
            print(f"Loaded model: {self.best_model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filepath}: {e}") from e

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the best model.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict, must match training feature columns

        Returns
        -------
        np.ndarray
            Predicted labels. If label_encoder is available, returns
            original class names; otherwise returns encoded integers.

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded
        ValueError
            If X is not a pandas DataFrame
        """
        if self.best_model is None:
            raise RuntimeError("No model has been trained yet. Train or load a model first.")
        
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        predictions = self.best_model.predict(X)
        
        # Convert back to class names if encoder available
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions

if __name__ == "__main__":
    # Example usage
    try:
        # Try relative imports when module is part of a package
        from .data_loader import DataLoader
        from .preprocessing import Preprocessor
    except Exception:
        # Fallback for running as a standalone script
        from data_loader import DataLoader
        from preprocessing import Preprocessor
    
    # Load and preprocess data
    loader = DataLoader('/files/project-MOISE/data')
    gtzan = loader.load_gtzan_data()
    
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test, features, le = preprocessor.preprocess_gtzan(gtzan)
    
    # Train and evaluate models
    classifier = GenreClassifier()
    results = classifier.train_and_evaluate(X_train, y_train, X_test, y_test, le)
    
    # Visualize results
    classifier.plot_model_comparison(results)

    # Obtain predictions from the trained best model and plot confusion matrix
    classifier.plot_confusion_matrix(y_test, results[classifier.best_model_name]['predictions'])
    
    # Save model
    classifier.save_model()