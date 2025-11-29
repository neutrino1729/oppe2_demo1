"""
Data Validation Tests
Tests data quality and schema compliance
"""

import pandas as pd
import pytest
import os

@pytest.fixture
def data():
    """Pytest fixture to load the transaction dataset"""
    data_path = "data/transactions.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)

def test_no_missing_values(data):
    """Tests that there are no null values in the dataset"""
    missing_count = data.isnull().sum().sum()
    assert missing_count == 0, f"Dataset contains {missing_count} missing values"

def test_core_columns_exist(data):
    """Tests that all expected columns are present"""
    # Core columns that must exist
    expected_cols = {'Time', 'Amount', 'Class'}
    # V1-V28 columns
    expected_cols.update({f'V{i}' for i in range(1, 29)})
    
    # Check if expected columns are present
    assert expected_cols.issubset(set(data.columns)), \
        f"Dataset is missing core columns. Missing: {expected_cols - set(data.columns)}"

def test_target_column_binary(data):
    """Tests that the target 'Class' column is binary (0 or 1)"""
    unique_classes = set(data["Class"].unique())
    assert unique_classes == {0, 1}, \
        f"Target column should only contain 0 and 1, found: {unique_classes}"

def test_amount_column_positive(data):
    """Tests that Amount column contains only non-negative values"""
    min_amount = data["Amount"].min()
    assert min_amount >= 0, f"Amount column contains negative values: min={min_amount}"

def test_dataset_not_empty(data):
    """Tests that dataset has data"""
    assert len(data) > 0, "Dataset is empty"

def test_dataset_size_reasonable(data):
    """Tests that dataset size is within expected range"""
    assert len(data) > 1000, f"Dataset too small: {len(data)} rows"
    assert len(data) < 1000000, f"Dataset too large: {len(data)} rows"

def test_fraud_rate_reasonable(data):
    """Tests that fraud rate is within reasonable bounds"""
    fraud_rate = data["Class"].mean()
    assert 0.001 <= fraud_rate <= 0.5, \
        f"Fraud rate {fraud_rate:.2%} is outside reasonable range (0.1% - 50%)"

def test_time_column_sorted(data):
    """Tests that Time column is sorted (transactions in order)"""
    is_sorted = data["Time"].is_monotonic_increasing
    assert is_sorted, "Time column should be sorted in ascending order"

def test_no_duplicate_rows(data):
    """Tests that there are no completely duplicate rows"""
    duplicate_count = data.duplicated().sum()
    duplicate_rate = duplicate_count / len(data)
    assert duplicate_rate < 0.01, \
        f"Too many duplicate rows: {duplicate_count} ({duplicate_rate:.2%})"
