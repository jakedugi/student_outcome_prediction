"""Tests for model classes."""
import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import MagicMock, patch
from sklearn.exceptions import NotFittedError

from src.models.sklearn_wrappers import (
    _wrap,
    DecisionTree,
    RandomForest,
    LogisticReg,
    SVM,
    KNN,
    XGBoost
)
from src.models.base import BaseClassifier

# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = pd.DataFrame({
        'feature1': np.random.random(100),
        'feature2': np.random.random(100),
        'feature3': np.random.random(100)
    })
    y = pd.Series(np.random.randint(0, 3, 100))
    return X, y

@pytest.fixture
def mock_estimator():
    """Create a mock sklearn estimator."""
    estimator = MagicMock()
    estimator.fit = MagicMock(return_value=estimator)
    estimator.predict = MagicMock(return_value=np.array([0, 1, 2] * 33 + [0]))  # 100 predictions
    estimator.predict_proba = MagicMock(return_value=np.array([[0.8, 0.1, 0.1]] * 100))
    return estimator

# Test base classifier
class TestBaseClassifier:
    def test_evaluate(self, sample_data):
        """Test the evaluate method of BaseClassifier."""
        X, y = sample_data
        
        class TestModel(BaseClassifier):
            def fit(self, X, y):
                return self
                
            def predict(self, X):
                # Return array of same length as input
                preds = np.array([0, 1, 2] * (len(X) // 3) + [0] * (len(X) % 3))
                self._last_predictions = preds  # Store predictions
                return preds
        
        model = TestModel()
        results = model.evaluate(X, y)
        
        assert "accuracy" in results
        assert "confusion_matrix" in results
        assert "y_pred" in results
        assert "model" in results
        assert "model_obj" in results
        
    def test_get_last_predictions(self, sample_data):
        """Test getting last predictions."""
        X, y = sample_data
        
        class TestModel(BaseClassifier):
            def fit(self, X, y):
                return self
                
            def predict(self, X):
                # Return array of same length as input
                preds = np.array([0, 1, 2] * (len(X) // 3) + [0] * (len(X) % 3))
                self._last_predictions = preds  # Store predictions
                return preds
        
        model = TestModel()
        
        # Should raise error before any predictions
        with pytest.raises(ValueError):
            model.get_last_predictions()
            
        # Should work after predict
        predictions = model.predict(X)
        last_predictions = model.get_last_predictions()
        assert last_predictions is not None
        assert isinstance(last_predictions, np.ndarray)
        assert len(last_predictions) == len(X)
        np.testing.assert_array_equal(last_predictions, predictions)

# Test sklearn wrappers
class TestSklearnWrappers:
    def test_wrap_factory(self):
        """Test the wrapper factory function."""
        MockEstimator = MagicMock()
        WrappedModel = _wrap("test_model", MockEstimator, param1=1)
        
        assert WrappedModel.model_name == "test_model"
        assert issubclass(WrappedModel, BaseClassifier)
        
        model = WrappedModel(param2=2)
        assert model.estimator is not None
        
    def test_model_fit(self, sample_data, mock_estimator):
        """Test model fitting."""
        X, y = sample_data
        
        # Test with different model types
        models = [
            DecisionTree(),
            RandomForest(),
            LogisticReg(),
            KNN()
        ]
        
        for model in models:
            model.estimator = mock_estimator
            fitted_model = model.fit(X, y)
            assert fitted_model is model
            mock_estimator.fit.assert_called_with(X, y)
            
    def test_model_predict(self, sample_data, mock_estimator):
        """Test model prediction."""
        X, y = sample_data
        
        model = DecisionTree()
        model.estimator = mock_estimator
        
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        mock_estimator.predict.assert_called_with(X)
        
    def test_model_predict_proba(self, sample_data, mock_estimator):
        """Test probability predictions."""
        X, y = sample_data
        
        # Test model with predict_proba
        model = DecisionTree()
        model.estimator = mock_estimator
        proba = model.predict_proba(X)
        assert isinstance(proba, np.ndarray)
        mock_estimator.predict_proba.assert_called_with(X)
        
        # Test model without predict_proba
        model.estimator = MagicMock()
        del model.estimator.predict_proba
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)
            
    def test_model_call(self, sample_data, mock_estimator):
        """Test model callable interface for SHAP."""
        X, y = sample_data
        
        # Test with predict_proba
        model = DecisionTree()
        model.estimator = mock_estimator
        result = model(X)
        assert isinstance(result, np.ndarray)
        mock_estimator.predict_proba.assert_called_with(X)
        
        # Test without predict_proba
        model.estimator = MagicMock()
        del model.estimator.predict_proba
        model.estimator.predict.return_value = np.zeros(len(X))  # Return numpy array
        result = model(X)
        assert isinstance(result, np.ndarray)
        model.estimator.predict.assert_called_with(X)
        
    def test_warning_suppression(self, sample_data):
        """Test that warnings are properly suppressed."""
        X, y = sample_data
        
        def mock_fit_with_warning(*args, **kwargs):
            warnings.warn("Test warning", RuntimeWarning)
            return MagicMock()
            
        model = DecisionTree()
        model.estimator = MagicMock()
        model.estimator.fit = mock_fit_with_warning
        
        # Ensure warning is not filtered
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")  # Ensure all warnings are captured
            model.fit(X, y)
            assert len(record) > 0
            assert record[0].category == RuntimeWarning 