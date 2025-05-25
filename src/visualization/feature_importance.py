"""Feature importance visualization utilities."""
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from ..models.base import BaseClassifier
from ..utils import logger

def create_importance_plot(
    values: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a bar plot for feature importance values.
    
    Args:
        values: Array of importance values
        feature_names: List of feature names
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    try:
        indices = np.argsort(values)[::-1]
        display_names = [str(name).replace('_', ' ').title() for name in feature_names]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(indices)), values[indices])
        
        # Add value labels on top of bars
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
        ax.set_title(title)
        plt.tight_layout()
        
        return fig, ax
    except Exception as e:
        logger.error(f"Failed to create importance plot: {e}")
        plt.close()  # Clean up in case of error
        raise

def get_feature_importance(
    model: BaseClassifier,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str],
    method: str = "auto"
) -> Optional[np.ndarray]:
    """Extract feature importance values using the specified method.
    
    Args:
        model: Model wrapper instance
        X: Input features
        feature_names: Feature names
        method: One of ["auto", "native", "shap", "coef"]
        
    Returns:
        Array of importance values or None if not available
    """
    try:
        estimator = model.estimator
        
        if method in ["auto", "native"] and hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
            
        if method in ["auto", "coef"] and hasattr(estimator, "coef_"):
            if isinstance(estimator.coef_, np.ndarray):
                return np.abs(estimator.coef_)
            return None
            
        if method in ["auto", "shap"]:
            try:
                explainer = shap.Explainer(estimator, X)
                shap_values = explainer(X)
                if isinstance(shap_values, shap.Explanation):
                    # Handle multiclass case
                    if len(shap_values.shape) > 2:
                        return np.abs(shap_values.values).mean(axis=(0, 2))
                    return np.abs(shap_values.values).mean(axis=0)
            except Exception as e:
                logger.warning(f"SHAP importance calculation failed: {e}")
                
        return None
    except Exception as e:
        logger.error(f"Failed to compute feature importance: {e}")
        return None

def plot_feature_importance(
    model: BaseClassifier,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    method: str = "auto",
    max_display: int = 20,
    class_descriptions: Optional[Dict[int, str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot feature importance using the most appropriate method.
    
    Args:
        model: Model wrapper instance
        X: Input features
        feature_names: Feature names (if None, will try to get from X)
        method: Importance calculation method
        max_display: Maximum features to display
        class_descriptions: Class label descriptions
        output_path: Optional path to save the plot
    """
    try:
        # Get feature names from DataFrame if not provided
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Convert X to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        importance_values = get_feature_importance(model, X_array, feature_names, method)
        
        if importance_values is not None:
            fig, ax = create_importance_plot(
                importance_values[:max_display],
                feature_names,
                title=f"Feature Importance - {model.model_name}"
            )
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {output_path}")
            else:
                plt.show()
        else:
            logger.warning(
                f"Could not compute feature importance for {model.model_name} "
                f"using method '{method}'"
            )
    except Exception as e:
        logger.error(f"Failed to plot feature importance: {e}")
        raise
    finally:
        plt.close()  # Always clean up 