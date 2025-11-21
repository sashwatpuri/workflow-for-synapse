"""
Machine Learning Model Training Module
Smart Farming Prediction System

This module implements:
1. Logistic Regression (probabilistic baseline)
2. Random Forest (main model)
3. XGBoost (optimized model)

All models output probabilities for P(irrigation_needed)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import joblib
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Main class for training and evaluating ML models"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        self.metrics = {}
        
    def train_logistic_regression(self, X_train, y_train, **kwargs) -> LogisticRegression:
        """
        Train Logistic Regression model (probabilistic baseline)
        
        Advantages:
        - Fast training
        - Interpretable coefficients
        - Outputs calibrated probabilities
        - Good baseline for comparison
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for LogisticRegression
        """
        print("\n" + "="*70)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*70)
        
        # Default parameters
        params = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced',  # Handle class imbalance
            'solver': 'lbfgs',
            'C': 1.0  # Regularization strength
        }
        params.update(kwargs)
        
        # Train model
        print(f"Parameters: {params}")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        print(f"Training complete!")
        print(f"Number of iterations: {model.n_iter_[0]}")
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest model (main model)
        
        Advantages:
        - Handles non-linear relationships
        - Feature importance
        - Robust to outliers
        - Good generalization
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for RandomForestClassifier
        """
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        
        # Default parameters
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1,  # Use all CPU cores
            'verbose': 0
        }
        params.update(kwargs)
        
        # Train model
        print(f"Parameters: {params}")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"Training complete!")
        print(f"Number of trees: {model.n_estimators}")
        print(f"Max depth: {model.max_depth}")
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs) -> xgb.XGBClassifier:
        """
        Train XGBoost model (optimized model)
        
        Advantages:
        - State-of-the-art performance
        - Handles missing values
        - Built-in regularization
        - Fast training with GPU support
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for XGBClassifier
        """
        print("\n" + "="*70)
        print("TRAINING XGBOOST")
        print("="*70)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Default parameters
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        params.update(kwargs)
        
        # Train model
        print(f"Parameters: {params}")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"Training complete!")
        print(f"Number of boosting rounds: {model.n_estimators}")
        print(f"Learning rate: {model.learning_rate}")
        
        self.models['xgboost'] = model
        return model
    
    def predict(self, model_name: str, X_test) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with a trained model
        
        Args:
            model_name: Name of the model
            X_test: Test features
            
        Returns:
            (predictions, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities (for positive class)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        self.predictions[model_name] = y_pred
        self.probabilities[model_name] = y_prob
        
        return y_pred, y_prob
    
    def evaluate(self, model_name: str, y_test, y_pred=None, y_prob=None) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model_name: Name of the model
            y_test: True labels
            y_pred: Predicted labels (optional, will use stored if None)
            y_prob: Predicted probabilities (optional, will use stored if None)
            
        Returns:
            Dictionary of metrics
        """
        if y_pred is None:
            y_pred = self.predictions.get(model_name)
        if y_prob is None:
            y_prob = self.probabilities.get(model_name)
        
        if y_pred is None or y_prob is None:
            raise ValueError(f"No predictions found for '{model_name}'")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'] = cm[0, 0]
        metrics['fp'] = cm[0, 1]
        metrics['fn'] = cm[1, 0]
        metrics['tp'] = cm[1, 1]
        
        # Specificity and sensitivity
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        
        self.metrics[model_name] = metrics
        return metrics
    
    def print_evaluation(self, model_name: str):
        """Print evaluation metrics"""
        if model_name not in self.metrics:
            print(f"No metrics found for '{model_name}'")
            return
        
        metrics = self.metrics[model_name]
        
        print(f"\n{'='*70}")
        print(f"EVALUATION: {model_name.upper()}")
        print(f"{'='*70}")
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['tn']:4d}  FP: {metrics['fp']:4d}")
        print(f"  FN: {metrics['fn']:4d}  TP: {metrics['tp']:4d}")
    
    def get_feature_importance(self, model_name: str, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For logistic regression, use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            print(f"Model '{model_name}' does not support feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, model_name: str, feature_names: List[str], top_n: int = 20):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(model_name, feature_names, top_n)
        
        if importance_df.empty:
            return
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'results/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTop {top_n} Features for {model_name}:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.6f}")
    
    def plot_roc_curve(self, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name in self.probabilities.keys():
            y_prob = self.probabilities[model_name]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = self.metrics[model_name]['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_test):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name in self.probabilities.keys():
            y_prob = self.probabilities[model_name]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            
            plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self, output_dir: str = 'models'):
        """Save all trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(output_dir, f'{model_name}.pkl')
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a trained model"""
        model = joblib.load(filepath)
        self.models[model_name] = model
        print(f"Loaded {model_name} from {filepath}")
        return model
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        if not self.metrics:
            print("No models evaluated yet")
            return pd.DataFrame()
        
        comparison = []
        for model_name, metrics in self.metrics.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        return comparison_df


def train_all_models():
    """Main function to train all models"""
    from data_preprocessing import DataPreprocessor
    import os
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and preprocess data
    print("="*70)
    print("SMART FARMING MODEL TRAINING PIPELINE")
    print("="*70)
    
    preprocessor = DataPreprocessor(
        r"c:\Users\sashwat puri sachdev\OneDrive\Documents\synapse pro project\Smart_Farming_Crop_Yield_2024.csv"
    )
    
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.run_full_pipeline()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    trainer.train_logistic_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_xgboost(X_train, y_train)
    
    # Make predictions and evaluate
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        trainer.predict(model_name, X_test)
        trainer.evaluate(model_name, y_test)
        trainer.print_evaluation(model_name)
        
        # Feature importance
        if model_name in ['random_forest', 'xgboost']:
            trainer.plot_feature_importance(model_name, feature_cols, top_n=20)
    
    # Compare models
    trainer.compare_models()
    
    # Plot ROC and PR curves
    trainer.plot_roc_curve(y_test)
    trainer.plot_precision_recall_curve(y_test)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print("Models saved to: models/")
    print("Results saved to: results/")
    
    return trainer, preprocessor


if __name__ == "__main__":
    trainer, preprocessor = train_all_models()
