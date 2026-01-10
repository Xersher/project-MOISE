"""
Popularity Prediction Module
============================

This module implements regression models to predict song popularity (streams)
based on Spotify audio features.

Models included:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor (with n_jobs=-1)
- Gradient Boosting Regressor
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class PopularityPredictor:
    """
    Class to build and evaluate popularity prediction models.

    This class provides methods to train multiple regression models,
    compare their performance, visualize results, and persist the best model.

    Attributes
    ----------
    model_directory : Path
        Directory to save trained models
    models : dict of str to sklearn estimator
        Dictionary of regression models
    best_model : sklearn estimator or None
        The best performing model after training
    best_model_name : str or None
        Name of the best model
    results : dict
        Training results for all models
    feature_names : list of str or None
        Names of features used in training
    """
    
    def __init__(self, model_directory: str = '/files/project-MOISE/results/models') -> None:
        """
        Initialize the PopularityPredictor.

        Parameters
        ----------
        model_directory : str, default='/files/project-MOISE/results/models'
            Directory to save trained models
        """
        self.model_directory: Path = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.results: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None
        
        print(f"PopularityPredictor initialized. Models will be saved to: {self.model_directory}")
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create multiple regression models to compare.

        Returns
        -------
        dict of str to sklearn estimator
            Dictionary mapping model names to regression model instances:
            - 'Linear Regression': LinearRegression
            - 'Ridge': Ridge with alpha=1.0
            - 'Lasso': Lasso with alpha=0.1
            - 'Random Forest': RandomForestRegressor with n_jobs=-1
            - 'Gradient Boosting': GradientBoostingRegressor
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=10000),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        print(f"Created {len(self.models)} models for comparison")
        return self.models
    
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
        ) -> Dict[str, Any]:
        """
        Train all models and evaluate their performance.

        For each model, computes R², MAE, RMSE, MAPE, and 5-fold
        cross-validation scores. Selects the best model based on
        the average of test R² and CV mean R².

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target (log-transformed streams)
        y_test : pd.Series
            Test target (log-transformed streams)

        Returns
        -------
        dict of str to dict
            Dictionary mapping model names to result dictionaries containing:
            - 'model': trained model instance
            - 'train_r2': float
            - 'test_r2': float
            - 'mae': float (Mean Absolute Error)
            - 'rmse': float (Root Mean Squared Error)
            - 'mape': float (Mean Absolute Percentage Error)
            - 'cv_mean': float (mean CV R²)
            - 'cv_std': float (CV standard deviation)
            - 'predictions': np.ndarray (test set predictions)

        Raises
        ------
        ValueError
            If input data is invalid (wrong type, empty, or mismatched lengths)
        RuntimeError
            If all models fail to train

        Notes
        -----
        Cross-validation uses n_jobs=-1 for parallel fold computation.
        """
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
        
        results = {}
        
        print("=" * 70)
        print("TRAINING AND EVALUATING POPULARITY PREDICTION MODELS")
        print("=" * 70)
        
        best_score = -np.inf # R² can be negative
        self.feature_names = list(X_train.columns)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate regression metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # MAPE (Mean Absolute Percentage Error)
                # Avoid division by zero
                mask = y_test != 0
                mape = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100
                
                # Cross-validation
                print("Running 5-fold cross-validation...")
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=5, 
                    scoring='r2',
                    n_jobs=-1
                )
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred_test
                }
                
                # Track best model using combined score (test + CV)
                combined_score = (test_r2 + cv_scores.mean()) / 2
                if combined_score > best_score:
                    best_score = combined_score
                    self.best_model = model
                    self.best_model_name = model_name
                
                # Print results
                print(f"Training R²: {train_r2:.4f}")
                print(f"Test R²: {test_r2:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAPE: {mape:.2f}%")
                print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                print(f"  Skipping {model_name}...")
                continue
        
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {self.best_model_name} (R² = {best_score:.4f})")
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
        1. R² score comparison across models
        2. MAE comparison across models

        Parameters
        ----------
        results : dict
            Results dictionary from train_and_evaluate
        save_path : Path or str, optional
            Path to save the figure. If None, saves to
            '{model_directory}/../figures/10_popularity_model_comparison.png'
        show : bool, default=False
            Whether to display the plot interactively

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """
        if save_path is None:
            save_path = self.model_directory.parent / 'figures' / '10_popularity_model_comparison.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_names = list(results.keys())
        test_r2s = [results[m]['test_r2'] for m in model_names]
        maes = [results[m]['mae'] for m in model_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # R² comparison
        axes[0].bar(model_names, test_r2s, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Model R² Score Comparison', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # MAE comparison
        axes[1].bar(model_names, maes, color='coral', alpha=0.7)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Model MAE Comparison', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Model comparison plot saved to {save_path}")

        return fig
    
    def plot_predictions(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[Path]=None,
        show: bool=False
        ) -> plt.Figure:
        """
        Plot actual vs predicted values.

        Creates a two-panel figure showing:
        1. Scatter plot of actual vs predicted with perfect prediction line
        2. Residual plot

        Parameters
        ----------
        y_test : pd.Series
            True target values
        y_pred : np.ndarray
            Predicted values from the model
        save_path : Path or str, optional
            Path to save the figure. If None, saves to
            '{model_directory}/../figures/11_popularity_predictions.png'
        show : bool, default=False
            Whether to display the plot interactively

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """
        if save_path is None:
            save_path = self.model_directory.parent / 'figures' / '11_popularity_predictions.png'
        else:
            save_path = Path(save_path)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.5, color='blue')
        axes[0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values (Log Streams)')
        axes[0].set_ylabel('Predicted Values (Log Streams)')
        axes[0].set_title(f'Actual vs Predicted - {self.best_model_name}', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, color='green')
        axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot', fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Predictions plot saved to {save_path}")

        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        top_n: int=15,
        save_path: Optional[Path]=None,
        show: bool=False
        ) -> plt.Figure:
        """
        Plot feature importance for tree-based models.

        Parameters
        ----------
        feature_names : list of str
            Names of features used in training
        top_n : int, default=15
            Number of top features to display
        save_path : Path or str, optional
            Path to save the figure. If None, saves to
            '{model_directory}/../figures/12_feature_importance.png'
        show : bool, default=False
            Whether to display the plot interactively

        Returns
        -------
        matplotlib.figure.Figure or None
            The generated figure object, or None if the best model
            doesn't support feature importance

        Notes
        -----
        Only works for Random Forest and Gradient Boosting models.
        """
        if save_path is None:
            save_path = self.model_directory.parent / 'figures' / '12_feature_importance.png'
        else:
            save_path = Path(save_path)

        if self.best_model_name not in ['Random Forest', 'Gradient Boosting']:
            print(f"[WARNING] {self.best_model_name} doesn't support feature importance")
            return
        n_features = len(feature_names)
        n_display = min(top_n, n_features)

        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig = plt.figure(figsize=(10, 8))
        plt.barh(range(n_display), importances[indices], color='darkgreen')
        plt.yticks(range(n_display), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {n_display} Features for Popularity Prediction', fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Feature importance plot saved to {save_path}")

        return fig
    
    def analyze_key_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        save_path: Optional[Path]=None,
        show: bool=False
        ) -> plt.Figure:
        """
        Analyze the relationship between key features and popularity.

        Creates scatter plots for the top 4 features most correlated
        with the target variable.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        feature_names : list of str
            Names of features
        save_path : Path or str, optional
            Path to save the figure. If None, saves to
            '{model_directory}/../figures/13_key_feature_analysis.png'
        show : bool, default=False
            Whether to display the plot interactively

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """
        if save_path is None:
            save_path = self.model_directory.parent / 'figures' / '13_key_feature_analysis.png'
        else:
            save_path = Path(save_path)

        # Select top features based on correlation
        X_df = pd.DataFrame(X, columns=feature_names)
        correlations = X_df.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
        top_features = correlations.head(4).index.tolist()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(top_features):
            axes[i].scatter(X_df[feature], y, alpha=0.3)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Popularity (Log Streams)')
            axes[i].set_title(f'{feature} vs Popularity', fontweight='bold')
            axes[i].grid(alpha=0.3)
            
            # Add correlation value
            corr = correlations[feature]
            axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                        transform=axes[i].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"✓ Key features analysis saved to {save_path}")

        return fig
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save the best trained model using joblib serialization.

        Saves the model along with metadata including model name,
        feature names, and training results for reproducibility.

        Parameters
        ----------
        filename : str, optional
            Custom filename (without path). If None, uses
            'best_popularity_predictor_{model_name}.pkl'

        Returns
        -------
        Path
            Path to the saved model file

        Raises
        ------
        RuntimeError
            If no model has been trained yet

        Notes
        -----
        The saved file contains a dictionary with keys:
        - 'model': the trained sklearn estimator
        - 'model_name': string identifier
        - 'feature_names': list of feature column names
        - 'results': full training results dictionary
        """
        if self.best_model is None:
            raise RuntimeError("No model has been trained yet!")

        if filename is None:
            filename = f"best_popularity_predictor_{self.best_model_name.replace(' ', '_').lower()}.pkl"

        filepath = self.model_directory / filename

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'results': self.results
        }

        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model.

        Restores the model, model name, feature names, and results
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
            self.feature_names = data.get('feature_names')
            self.results = data.get('results', {})

            print(f"✓ Model loaded from {filepath}")
            print(f"  Loaded model: {self.best_model_name}")

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
            Predicted popularity values (log-transformed if trained on log streams)

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded
        ValueError
            If X is not a pandas DataFrame
        """
        if self.best_model is None:
            raise RuntimeError("No model has been trained or loaded yet!")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        return self.best_model.predict(X)


if __name__ == "__main__":
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
    spotify = loader.load_spotify_data()
    
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_spotify(spotify)
    
 