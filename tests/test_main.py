import pytest
import subprocess
import sys
import time
import os
import pandas as pd
import numpy as np
from src.config import DATA_DIR, DEFAULT_CSV, NUMERIC_COLUMNS, TARGET

def test_main_script_execution(tmp_path):
    """Test that main.py executes successfully within time limit."""
    # Create dummy dataset
    DATA_DIR.mkdir(exist_ok=True)
    n_samples = 100
    data = {
        'student_id': range(n_samples),
        TARGET: np.random.choice([0, 1], size=n_samples)
    }
    
    # Add numeric columns
    for col in NUMERIC_COLUMNS:
        if 'grade' in col.lower():
            data[col] = np.random.uniform(0, 20, n_samples)
        elif 'age' in col.lower():
            data[col] = np.random.uniform(18, 60, n_samples)
        else:
            data[col] = np.random.uniform(0, 10, n_samples)
            
    df = pd.DataFrame(data)
    df.to_csv(DEFAULT_CSV, index=False)
    
    start_time = time.time()
    
    # Run main.py with minimal data (0 semesters)
    process = subprocess.run(
        [sys.executable, "main.py", "train", "--semesters", "0"],
        capture_output=True,
        text=True
    )
    
    execution_time = time.time() - start_time
    
    # Clean up
    if os.path.exists(DEFAULT_CSV):
        os.remove(DEFAULT_CSV)
    
    # Check process completed successfully
    assert process.returncode == 0
    
    # Check execution time is under 90 seconds (allowing for initialization overhead)
    assert execution_time < 90, f"Execution took {execution_time:.1f}s > 90s limit" 