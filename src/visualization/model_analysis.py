"""Model analysis and visualization utilities."""
from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import shap
from ..models.base import BaseClassifier
from ..utils import logger
from .feature_importance import (
    get_feature_importance,
    create_importance_plot,
    plot_feature_importance as plot_feature_importance_base
)

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
    """Plot feature importance using either native feature_importances_, coefficients, or SHAP values.
    
    This is a convenience wrapper around the base plot_feature_importance function
    that adds class-specific visualizations for multiclass models.
    
    Args:
        model_wrapper: Our custom model wrapper (BaseClassifier instance)
        X: Feature matrix to explain
        feature_names: List of feature names
        max_display: Maximum number of features to display
        class_descriptions: Optional mapping of class indices to descriptions
    """
    # Use default class descriptions if none provided
    class_desc = class_descriptions or CLASS_DESCRIPTIONS
    
    # Try to get SHAP values for multiclass visualization
    try:
        explainer = shap.Explainer(model_wrapper.estimator, X)
        shap_values = explainer(X)
        
        if isinstance(shap_values, shap.Explanation) and len(shap_values.shape) > 2:
            # Multiclass case - plot per-class importance
            for class_idx in range(shap_values.shape[2]):
                class_name = class_desc.get(class_idx, f"Class {class_idx}")
                plt.figure(figsize=(12, 8))
                
                # Use SHAP's summary plot for this class
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    X,
                    feature_names=feature_names,
                    plot_type="bar",
                    max_display=max_display,
                    show=False
                )
                
                plt.title(f"Feature Impact on {class_name} Outcome\n{model_wrapper.model_name}")
                
                # Add legend explaining SHAP values
                plt.figtext(
                    1.02, 0.5,
                    'How to read this plot:\n\n' +
                    '• Longer bars = Stronger impact\n' +
                    '• Red = Higher feature values\n' +
                    '• Blue = Lower feature values\n' +
                    '• Values show average impact\n' +
                    '  on model predictions',
                    fontsize=10, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                plt.tight_layout()
                plt.show()
            return
            
    except Exception as e:
        logger.debug(f"Could not compute class-specific SHAP values: {e}")
    
    # Fall back to standard feature importance plot
    plot_feature_importance_base(
        model_wrapper,
        X,
        feature_names,
        method="auto",
        max_display=max_display,
        class_descriptions=class_descriptions
    ) 