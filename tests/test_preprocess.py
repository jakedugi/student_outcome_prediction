import pytest
import pandas as pd
import numpy as np
from src.preprocess import Preprocessor
from src.config import NUMERIC_COLUMNS, TARGET

def test_preprocessor_output():
    """Test that preprocessor produces expected output shapes and no NaNs."""
    # Create dummy data
    n_samples = 100
    data = {
        'student_id': range(n_samples),
        TARGET: np.random.choice([0, 1], size=n_samples)
    }
    
    # Add numeric columns
    for col in NUMERIC_COLUMNS:
        if 'grade' in col.lower():
            data[col] = np.random.uniform(0, 20, n_samples)  # grades 0-20
        elif 'age' in col.lower():
            data[col] = np.random.uniform(18, 60, n_samples)  # age 18-60
        else:
            data[col] = np.random.uniform(0, 10, n_samples)  # other metrics
            
    raw_df = pd.DataFrame(data)
    
    # Run preprocessing
    preprocessor = Preprocessor()
    processed_df = preprocessor.fit_transform(raw_df)
    
    # Check that output is a DataFrame
    assert isinstance(processed_df, pd.DataFrame)
    
    # Check no NaN values
    assert not processed_df.isna().any().any()
    
    # Test semester features
    semester_df = preprocessor.semester_features(processed_df, semesters=2)
    assert isinstance(semester_df, pd.DataFrame)
    assert not semester_df.isna().any().any() 