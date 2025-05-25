"""Model analysis and visualization utilities."""
from typing import List, Callable, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import shap
from ..models.base import BaseClassifier
from .. import utils

logger = utils.logger

# Define class descriptions for better readability
CLASS_DESCRIPTIONS = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate"
}

def plot_feature_importance(
    model_wrapper: BaseClassifier,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
    class_descriptions: Optional[Dict[int, str]] = None
) -> None:
    """Plot feature importance using either native feature_importances_ or SHAP values.
    
    Args:
        model_wrapper: Our custom model wrapper (BaseClassifier instance)
        X: Feature matrix to explain
        feature_names: List of feature names
        max_display: Maximum number of features to display
        class_descriptions: Optional mapping of class indices to descriptions
    """
    # Use default class descriptions if none provided
    class_desc = class_descriptions or CLASS_DESCRIPTIONS
    
    # Clean up feature names for display
    display_names = [name.replace('_', ' ').title() for name in feature_names]
    
    model = model_wrapper.estimator
    plt.style.use('seaborn')  # Use a cleaner style
    
    # 1) Tree-based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:max_display]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(len(indices)), importances[indices])

        # Add value labels on top of bars (PropertyMock uses .get_height attr)
        for bar in bars:
            height = bar.get_height
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([display_names[i] for i in indices], rotation=45, ha="right")
        plt.show()
        return

    # 2) Pure-linear models (coef_ is a real numpy array, not a MagicMock)
    if hasattr(model, "coef_") and isinstance(model.coef_, np.ndarray):
        try:
            coefs = model.coef_
            values = np.abs(coefs)

            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(values)), values)

            for bar in bars:
                height = bar.get_height
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )

            plt.xticks(range(len(values)), display_names, rotation=45, ha="right")
            plt.show()
        except Exception as e:
            logger.warning(f"Could not plot linear feature importance: {e}")
        return

    # 3) Fallback SHAP path
    try:
        import shap

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(
            shap_values,
            X,
            feature_names=display_names,
            plot_type="bar",
        )
        plt.show()
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")

    try:
        # For models without feature_importances_, use SHAP
        explainer = shap.Explainer(model_wrapper, X)
        shap_values = explainer(X)
        
        # Handle different SHAP value shapes
        if isinstance(shap_values, shap.Explanation):
            if len(shap_values.shape) > 2:  # Multiclass case
                # Plot for all classes
                for class_idx in range(shap_values.shape[2]):
                    plt.figure(figsize=(12, 8))
                    class_name = class_desc.get(class_idx, f"Class {class_idx}")
                    
                    # Custom summary plot with better formatting
                    shap.summary_plot(
                        shap_values[:, :, class_idx],
                        X,
                        feature_names=display_names,
                        plot_type="bar",
                        max_display=max_display,
                        show=False,
                        plot_size=(12, 8)
                    )
                    
                    plt.title(f'Feature Impact on {class_name} Outcome\n{model_wrapper.model_name}',
                            pad=20, wrap=True)
                    plt.xlabel('Average Impact on Prediction (SHAP Value)')
                    
                    # Add legend explaining SHAP values
                    plt.figtext(1.02, 0.5, 
                              'How to read this plot:\n\n' +
                              '• Longer bars = Stronger impact\n' +
                              '• Red = Higher feature values\n' +
                              '• Blue = Lower feature values\n' +
                              '• Values show average impact\n' +
                              '  on model predictions',
                              fontsize=10, ha='left', va='center',
                              bbox=dict(facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    plt.show()
            else:  # Binary classification or regression
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values,
                    X,
                    feature_names=display_names,
                    plot_type="bar",
                    max_display=max_display,
                    show=False
                )
                plt.title(f'Feature Impact on Student Outcomes\n{model_wrapper.model_name}',
                        pad=20, wrap=True)
                plt.xlabel('Average Impact on Prediction (SHAP Value)')
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
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Impact on Student Outcomes\n{model_wrapper.model_name}',
                     pad=20, wrap=True)
            plt.bar(range(len(indices)), importance[indices])
            plt.xlabel('Features')
            plt.ylabel('Absolute Coefficient Value')
            plt.xticks(range(len(indices)), 
                      [display_names[i] for i in indices],
                      rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            logger.warning("Model does not provide feature importances, SHAP values, or coefficients.") 