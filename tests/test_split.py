import pytest
import pandas as pd
import numpy as np
from src.split import make_split
from src.config import TARGET

def test_split_proportions():
    """Test that split proportions match expected values."""
    # Create dummy data
    n_samples = 1000
    df = pd.DataFrame({
        TARGET: np.random.choice([0, 1], size=n_samples),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })
    
    # Perform split
    X_train, X_test, y_train, y_test = make_split(df)
    
    # Check split sizes (default test_size=0.2)
    assert len(X_train) == int(n_samples * 0.8)
    assert len(X_test) == int(n_samples * 0.2)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    
    # Check that no data was lost
    assert len(X_train) + len(X_test) == n_samples 