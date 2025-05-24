import pytest
import subprocess
import sys
import time

def test_main_script_execution():
    """Test that main.py executes successfully within time limit."""
    start_time = time.time()
    
    # Run main.py with minimal data (0 semesters)
    process = subprocess.run(
        [sys.executable, "main.py", "train", "--semesters", "0"],
        capture_output=True,
        text=True
    )
    
    execution_time = time.time() - start_time
    
    # Check process completed successfully
    assert process.returncode == 0
    
    # Check execution time is under 90 seconds (allowing for initialization overhead)
    assert execution_time < 90, f"Execution took {execution_time:.1f}s > 90s limit" 