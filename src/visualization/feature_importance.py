"""Feature importance visualization using SHAP values."""
from __future__ import annotations
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

from ..models.base import BaseClassifier
from .. import config, utils

logger = utils.logger

def plot_feature_importance(
    model: BaseClassifier,
    X: pd.DataFrame,
    output_path: Optional[Path] = None,
    max_display: int = 20
) -> None:
    """Generate and save SHAP feature importance plot.
    
    Args:
        model: Trained model that implements predict
        X: Feature matrix to explain
        output_path: Where to save the plot (if None, just displays)
        max_display: Maximum number of features to show
    """
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model.estimator, X)
        shap_values = explainer(X)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        
        # Add title and adjust layout
        plt.title("Feature Importance (SHAP Values)", pad=20)
        plt.tight_layout()
        
        # Save or display
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info("Feature importance plot saved to %s", output_path)
        else:
            plt.show()
            
    except Exception as e:
        logger.error("Failed to generate feature importance plot: %s", str(e))
        raise
    finally:
        plt.close()  # Clean up 