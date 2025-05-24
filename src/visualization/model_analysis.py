"""Model analysis and visualization utilities."""
from typing import List, Callable
import numpy as np
import matplotlib.pyplot as plt
import shap
from ..models.base import BaseClassifier

def plot_feature_importance(
    model_wrapper: BaseClassifier,
    X: np.ndarray,
    feature_names: List[str]
) -> None:
    """Plot feature importance using either native feature_importances_ or SHAP values.
    
    Args:
        model_wrapper: Our custom model wrapper (BaseClassifier instance)
        X: Feature matrix to explain
        feature_names: List of feature names
    """
    # Get the underlying scikit-learn model
    model = model_wrapper.estimator
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, XGBoost, etc.)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances ({model_wrapper.model_name})')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        # For other models, use SHAP values
        # For models without feature_importances_, we pass their predict method
        predict_fn: Callable = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict
        
        try:
            explainer = shap.Explainer(predict_fn, X)
            shap_values = explainer(X)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False,
                            plot_title=f'SHAP Values ({model_wrapper.model_name})')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not compute SHAP values for {model_wrapper.model_name}: {str(e)}")
            print("Falling back to coefficients if available...")
            
            # Fallback for linear models
            if hasattr(model, 'coef_'):
                coef = model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
                importance = np.abs(coef)
                indices = np.argsort(importance)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Coefficients ({model_wrapper.model_name})')
                plt.bar(range(X.shape[1]), importance[indices])
                plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
                plt.tight_layout()
                plt.show() 