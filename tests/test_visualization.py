"""Tests for visualization modules."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from src.visualization import model_analysis, feature_importance
from src.models.base import BaseClassifier

# Fixtures
@pytest.fixture
def mock_model():
    """Create a mock model with different types of feature importance."""
    model = MagicMock(spec=BaseClassifier)
    model.model_name = "TestModel"
    return model

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = pd.DataFrame({
        'feature1': np.random.random(100),
        'feature2': np.random.random(100),
        'feature3': np.random.random(100)
    })
    return X

@pytest.fixture
def feature_names():
    """Sample feature names."""
    return ['feature1', 'feature2', 'feature3']

# Test model_analysis.py
class TestModelAnalysis:
    def test_plot_feature_importance_tree_based(self, mock_model, sample_data, feature_names):
        """Test plotting feature importance for tree-based models."""
        # Setup tree-based model mock
        mock_model.estimator = MagicMock()
        mock_model.estimator.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        with patch('matplotlib.pyplot.show'):
            model_analysis.plot_feature_importance(
                mock_model,
                sample_data.values,
                feature_names
            )
            
    def test_plot_feature_importance_shap_multiclass(self, mock_model, sample_data, feature_names):
        """Test plotting SHAP importance for multiclass models."""
        # Setup SHAP mock
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.shape = (100, 3, 3)  # (samples, features, classes)
        
        with patch('shap.Explainer', return_value=mock_explainer) as mock_shap:
            with patch('matplotlib.pyplot.show'):
                mock_explainer.return_value = mock_shap_values
                
                model_analysis.plot_feature_importance(
                    mock_model,
                    sample_data.values,
                    feature_names
                )
                
    def test_plot_feature_importance_linear(self, mock_model, sample_data, feature_names):
        """Test plotting feature importance for linear models."""
        # Setup linear model mock
        mock_model.estimator = MagicMock()
        mock_model.estimator.coef_ = np.array([0.5, 0.3, 0.2])
        
        # Mock SHAP to fail
        with patch('shap.Explainer', side_effect=Exception("SHAP failed")):
            with patch('matplotlib.pyplot.show'):
                model_analysis.plot_feature_importance(
                    mock_model,
                    sample_data.values,
                    feature_names
                )

    def test_plot_feature_importance_no_importance(self, mock_model, sample_data, feature_names):
        """Test handling when no importance method is available."""
        mock_model.estimator = MagicMock()
        
        # Mock SHAP to fail
        with patch('shap.Explainer', side_effect=Exception("SHAP failed")):
            with patch('matplotlib.pyplot.show'):
                with patch('src.utils.logger.warning') as mock_warning:
                    model_analysis.plot_feature_importance(
                        mock_model,
                        sample_data.values,
                        feature_names
                    )
                    mock_warning.assert_called()

# Test feature_importance.py
class TestFeatureImportance:
    def test_plot_feature_importance_display(self, mock_model, sample_data):
        """Test feature importance plot display."""
        with patch('shap.Explainer') as mock_explainer:
            with patch('shap.summary_plot'):
                with patch('matplotlib.pyplot.show'):
                    feature_importance.plot_feature_importance(
                        mock_model,
                        sample_data
                    )

    def test_plot_feature_importance_save(self, mock_model, sample_data, tmp_path):
        """Test feature importance plot saving."""
        output_path = tmp_path / "feature_importance.png"
        
        with patch('shap.Explainer') as mock_explainer:
            with patch('shap.summary_plot'):
                with patch('matplotlib.pyplot.savefig'):
                    feature_importance.plot_feature_importance(
                        mock_model,
                        sample_data,
                        output_path
                    )

    def test_plot_feature_importance_error(self, mock_model, sample_data):
        """Test error handling in feature importance plotting."""
        with patch('shap.Explainer', side_effect=Exception("SHAP error")):
            with pytest.raises(Exception):
                feature_importance.plot_feature_importance(
                    mock_model,
                    sample_data
                )

    def test_cleanup(self, mock_model, sample_data):
        """Test that plt.close is called even if there's an error."""
        with patch('matplotlib.pyplot.close') as mock_close:
            with patch('shap.Explainer', side_effect=Exception("SHAP error")):
                with pytest.raises(Exception):
                    feature_importance.plot_feature_importance(
                        mock_model,
                        sample_data
                    )
                mock_close.assert_called_once() 