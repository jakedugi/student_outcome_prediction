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
    XGBoost,
    ModelConfig
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
                
            def predict_proba(self, X):
                # Return dummy probabilities
                n_samples = len(X)
                return np.ones((n_samples, 3)) / 3  # Equal probabilities for 3 classes
        
        model = TestModel()
        results = model.evaluate(X, y)
        
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "confusion_matrix" in results
        assert "y_pred" in results
        assert len(results["y_pred"]) == len(y)
        
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
                
            def predict_proba(self, X):
                # Return dummy probabilities
                n_samples = len(X)
                return np.ones((n_samples, 3)) / 3
        
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
        config = ModelConfig(
            name="test_model",
            estimator_class=MagicMock,
            default_params={"param1": 1}
        )
        WrappedModel = _wrap(config)
        
        assert issubclass(WrappedModel, BaseClassifier)
        assert WrappedModel.model_name == "test_model"
        
        # Test instantiation
        model = WrappedModel()
        assert isinstance(model.estimator, MagicMock)
        assert model.estimator.param1 == 1
        
    def test_model_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        config = ModelConfig(
            name="test_model",
            estimator_class=MagicMock,
            default_params={}
        )
        Model = _wrap(config)
        model = Model()
        
        result = model.fit(X, y)
        assert result is model  # Should return self
        model.estimator.fit.assert_called_once_with(X, y)
        
    def test_model_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        config = ModelConfig(
            name="test_model",
            estimator_class=MagicMock,
            default_params={}
        )
        Model = _wrap(config)
        model = Model()
        
        # Configure mock
        expected = np.zeros(len(X))
        model.estimator.predict.return_value = expected
        
        result = model.predict(X)
        np.testing.assert_array_equal(result, expected)
        model.estimator.predict.assert_called_once_with(X)
        
    def test_model_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        config = ModelConfig(
            name="test_model",
            estimator_class=MagicMock,
            default_params={}
        )
        Model = _wrap(config)
        model = Model()
        
        # Test with probability support
        expected = np.random.random((len(X), 3))
        model.estimator.predict_proba.return_value = expected
        
        result = model.predict_proba(X)
        np.testing.assert_array_equal(result, expected)
        
        # Test without probability support
        del model.estimator.predict_proba
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)
            
    def test_model_call(self, sample_data):
        """Test model callable interface."""
        X, y = sample_data
        config = ModelConfig(
            name="test_model",
            estimator_class=MagicMock,
            default_params={}
        )
        Model = _wrap(config)
        model = Model()
        
        # Test with probability support
        expected = np.random.random((len(X), 3))
        model.estimator.predict_proba.return_value = expected
        
        result = model(X)
        np.testing.assert_array_equal(result, expected)
        
        # Test without probability support
        del model.estimator.predict_proba
        expected = np.zeros(len(X))
        model.estimator.predict.return_value = expected
        
        result = model(X)
        np.testing.assert_array_equal(result, expected)
        
    def test_warning_suppression(self, sample_data):
        """Test that warnings are properly suppressed."""
        X, y = sample_data
        config = ModelConfig(
            name="test_model",
            estimator_class=MagicMock,
            default_params={}
        )
        Model = _wrap(config)
        model = Model()
        
        def mock_fit_with_warning(*args, **kwargs):
            warnings.warn("Test warning", RuntimeWarning)
            return model.estimator
            
        model.estimator.fit = mock_fit_with_warning
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Make warnings raise errors
            # Should not raise warning
            model.fit(X, y) 