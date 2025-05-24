"""Model analysis and visualization utilities."""
from typing import List, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import shap
from ..models.base import BaseClassifier
from .. import utils

logger = utils.logger

def plot_feature_importance(
    model_wrapper: BaseClassifier,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 20
) -> None:
    """Plot feature importance using either native feature_importances_ or SHAP values.
    
    Args:
        model_wrapper: Our custom model wrapper (BaseClassifier instance)
        X: Feature matrix to explain
        feature_names: List of feature names
        max_display: Maximum number of features to display
    """
    model = model_wrapper.estimator
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, XGBoost, etc.)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:max_display]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances ({model_wrapper.model_name})')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        try:
            # For models without feature_importances_, use SHAP
            # The model wrapper is now callable and SHAP-compatible
            explainer = shap.Explainer(model_wrapper, X)
            shap_values = explainer(X)
            
            # Handle different SHAP value shapes
            if isinstance(shap_values, shap.Explanation):
                if len(shap_values.shape) > 2:  # Multiclass case
                    # Plot for all classes
                    for class_idx in range(shap_values.shape[2]):
                        plt.figure(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values[:, :, class_idx],
                            X,
                            feature_names=feature_names,
                            plot_type="bar",
                            max_display=max_display,
                            show=False
                        )
                        plt.title(f'SHAP Values - Class {class_idx} ({model_wrapper.model_name})')
                        plt.tight_layout()
                        plt.show()
                else:  # Binary classification or regression
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values,
                        X,
                        feature_names=feature_names,
                        plot_type="bar",
                        max_display=max_display,
                        show=False
                    )
                    plt.title(f'SHAP Values ({model_wrapper.model_name})')
                    plt.tight_layout()
                    plt.show()
            
        except Exception as e:
            logger.warning("Could not compute SHAP values: %s", str(e))
            logger.info("Falling back to coefficients if available...")
            
            # Fallback for linear models
            if hasattr(model, 'coef_'):
                coef = model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
                importance = np.abs(coef)
                indices = np.argsort(importance)[::-1][:max_display]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Coefficients ({model_wrapper.model_name})')
                plt.bar(range(len(indices)), importance[indices])
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            else:
                logger.warning("Model does not provide feature importances, SHAP values, or coefficients.") 