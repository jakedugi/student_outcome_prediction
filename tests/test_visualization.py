"""Tests for visualization modules."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, PropertyMock

from src.visualization import model_analysis, feature_importance
from src.models.base import BaseClassifier

# Fixtures
@pytest.fixture
def mock_model():
    """Create a mock model with different types of feature importance."""
    model = MagicMock(spec=BaseClassifier)
    model.model_name = "TestModel"
    model.estimator = MagicMock()  # Add estimator attribute
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
        # Setup tree-based model mock with proper numpy array
        importances = np.array([0.5, 0.3, 0.2])
        mock_model.estimator.feature_importances_ = importances
        
        with patch('matplotlib.pyplot.style.use'):  # Mock style.use
            with patch('matplotlib.pyplot.show'):
                with patch('matplotlib.pyplot.subplots') as mock_subplots:
                    # Mock the axes object
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (MagicMock(), mock_ax)
                    
                    # Mock bar plot return
                    mock_ax.bar.return_value = [MagicMock()]
                    
                    model_analysis.plot_feature_importance(
                        mock_model,
                        sample_data.values,
                        feature_names
                    )
            
    def test_plot_feature_importance_shap_multiclass(self, mock_model, sample_data, feature_names):
        """Test plotting SHAP importance for multiclass models."""
        # Setup SHAP mock with proper numpy arrays
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.shape = (100, 3, 3)  # (samples, features, classes)
        mock_shap_values.values = np.zeros((100, 3, 3))  # Add actual numpy array
        
        with patch('matplotlib.pyplot.style.use'):  # Mock style.use
            with patch('shap.Explainer', return_value=mock_explainer) as mock_shap:
                with patch('matplotlib.pyplot.show'):
                    with patch('shap.summary_plot'):
                        mock_explainer.return_value = mock_shap_values
                        
                        model_analysis.plot_feature_importance(
                            mock_model,
                            sample_data.values,
                            feature_names
                        )
                
    def test_plot_feature_importance_linear(self, mock_model, sample_data, feature_names):
        """Test plotting feature importance for linear models."""
        # Setup linear model mock with proper numpy array
        coef = np.array([0.5, 0.3, 0.2])
        mock_model.estimator.coef_ = coef
        
        with patch('matplotlib.pyplot.style.use'):  # Mock style.use
            # Mock SHAP to fail
            with patch('shap.Explainer', side_effect=Exception("SHAP failed")):
                with patch('matplotlib.pyplot.show'):
                    with patch('matplotlib.pyplot.figure'):
                        with patch('matplotlib.pyplot.bar') as mock_bar:
                            mock_bar.return_value = [MagicMock()]
                            
                            model_analysis.plot_feature_importance(
                                mock_model,
                                sample_data.values,
                                feature_names
                            )

    def test_plot_feature_importance_no_importance(self, mock_model, sample_data, feature_names):
        """Test handling when no importance method is available."""
        # Remove feature importances and coefficients
        mock_model.estimator = MagicMock()
        if hasattr(mock_model.estimator, 'feature_importances_'):
            delattr(mock_model.estimator, 'feature_importances_')
        if hasattr(mock_model.estimator, 'coef_'):
            delattr(mock_model.estimator, 'coef_')
        
        with patch('matplotlib.pyplot.style.use'):  # Mock style.use
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
        mock_model.estimator = MagicMock()  # Ensure estimator is set
        
        # Setup SHAP mock with proper numpy arrays
        mock_explainer = MagicMock()
        mock_values = np.zeros((len(sample_data), len(sample_data.columns)))
        mock_explainer.return_value = mock_values
        
        with patch('shap.Explainer', return_value=mock_explainer) as mock_shap:
            with patch('shap.summary_plot'):
                with patch('matplotlib.pyplot.show'):
                    feature_importance.plot_feature_importance(
                        mock_model,
                        sample_data
                    )

    def test_plot_feature_importance_save(self, mock_model, sample_data, tmp_path):
        """Test feature importance plot saving."""
        mock_model.estimator = MagicMock()  # Ensure estimator is set
        output_path = tmp_path / "feature_importance.png"
        
        # Setup SHAP mock with proper numpy arrays
        mock_explainer = MagicMock()
        mock_values = np.zeros((len(sample_data), len(sample_data.columns)))
        mock_explainer.return_value = mock_values
        
        with patch('shap.Explainer', return_value=mock_explainer) as mock_shap:
            with patch('shap.summary_plot'):
                with patch('matplotlib.pyplot.savefig'):
                    feature_importance.plot_feature_importance(
                        mock_model,
                        sample_data,
                        output_path
                    )

    def test_plot_feature_importance_error(self, mock_model, sample_data):
        """Test error handling in feature importance plotting."""
        mock_model.estimator = MagicMock()  # Ensure estimator is set
        
        with patch('shap.Explainer', side_effect=Exception("SHAP error")):
            with pytest.raises(Exception):
                feature_importance.plot_feature_importance(
                    mock_model,
                    sample_data
                )

    def test_cleanup(self, mock_model, sample_data):
        """Test that plt.close is called even if there's an error."""
        mock_model.estimator = MagicMock()  # Ensure estimator is set
        
        with patch('matplotlib.pyplot.close') as mock_close:
            with patch('shap.Explainer', side_effect=Exception("SHAP error")):
                with pytest.raises(Exception):
                    feature_importance.plot_feature_importance(
                        mock_model,
                        sample_data
                    )
                mock_close.assert_called_once() 