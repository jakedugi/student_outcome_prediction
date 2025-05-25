"""Shared test fixtures and configuration."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import shap
from unittest.mock import MagicMock
from typing import Generator, Tuple

from src.models.base import BaseClassifier
from src.config import Config, LoggingConfig, ModelConfig, VisualizationConfig

@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample training data.
    
    Returns:
        X: Feature DataFrame
        y: Target series
    """
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.random(100),
        'feature2': np.random.random(100),
        'feature3': np.random.random(100)
    })
    y = pd.Series(np.random.randint(0, 3, 100), name='target')
    return X, y

@pytest.fixture
def mock_model() -> Generator[MagicMock, None, None]:
    """Create a mock model with configurable behavior.
    
    Yields:
        Mock model instance
    """
    model = MagicMock(spec=BaseClassifier)
    model.model_name = "TestModel"
    model.estimator = MagicMock()
    
    # Configure default behavior
    model.estimator.feature_importances_ = np.array([0.5, 0.3, 0.2])
    model.estimator.predict.return_value = np.zeros(100)
    model.estimator.predict_proba.return_value = np.random.random((100, 3))
    
    yield model

@pytest.fixture
def mock_shap_values() -> Generator[MagicMock, None, None]:
    """Create mock SHAP values.
    
    Yields:
        Mock SHAP values
    """
    values = MagicMock(spec=shap.Explanation)
    values.shape = (100, 3, 3)  # (samples, features, classes)
    values.values = np.random.random((100, 3, 3))
    yield values

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary output directory.
    
    Args:
        tmp_path: Pytest temporary path fixture
        
    Yields:
        Path to output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir

@pytest.fixture
def test_config(temp_output_dir: Path) -> Config:
    """Create test configuration.
    
    Args:
        temp_output_dir: Temporary output directory
        
    Returns:
        Test configuration
    """
    return Config(
        logging=LoggingConfig(
            level="DEBUG",
            format="%(message)s",
            file=str(temp_output_dir / "test.log")
        ),
        model=ModelConfig(
            random_state=42,
            test_size=0.2,
            cv_folds=2
        ),
        viz=VisualizationConfig(
            style="default",
            dpi=72,
            save_format="png"
        ),
        data_dir=temp_output_dir,  # Use temp dir for data too
        output_dir=temp_output_dir
    )

@pytest.fixture(autouse=True)
def setup_test_env(test_config: Config) -> Generator[None, None, None]:
    """Set up test environment.
    
    This fixture runs automatically for all tests.
    
    Args:
        test_config: Test configuration
        
    Yields:
        None
    """
    # Configure logging
    test_config.setup_logging()
    
    yield
    
    # Clean up matplotlib
    import matplotlib.pyplot as plt
    plt.close('all') 