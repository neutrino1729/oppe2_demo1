"""
Model Performance Tests
Tests trained model quality and functionality
"""

import pandas as pd
import joblib
import os
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

@pytest.fixture
def model():
    """Load the trained model"""
    model_path = "artifacts/model.pkl"
    if not os.path.exists(model_path):
        pytest.skip(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)

@pytest.fixture
def test_data():
    """Load and prepare test data"""
    data_path = "data/transactions.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Get features (excluding target and non-features)
    columns_to_drop = ['Class', 'Time']
    if 'location' in df.columns:
        columns_to_drop.append('location')
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df['Class']
    
    # Create test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_test, y_test

def test_model_artifact_exists():
    """Tests if the model artifact exists"""
    assert os.path.exists("artifacts/model.pkl"), \
        "Model artifact 'model.pkl' not found"

def test_model_params_exist():
    """Tests if model parameters file exists"""
    assert os.path.exists("artifacts/params.json"), \
        "Model parameters 'params.json' not found"

def test_model_can_predict(model, test_data):
    """Tests that model can make predictions"""
    X_test, y_test = test_data
    
    # Should not raise any exceptions
    predictions = model.predict(X_test)
    
    # Check output shape
    assert len(predictions) == len(X_test), \
        "Prediction length doesn't match input length"

def test_model_predictions_binary(model, test_data):
    """Tests that model predictions are binary"""
    X_test, _ = test_data
    predictions = model.predict(X_test)
    
    unique_predictions = set(predictions)
    assert unique_predictions.issubset({0, 1}), \
        f"Model predictions should be 0 or 1, found: {unique_predictions}"

def test_model_f1_score_acceptable(model, test_data):
    """Tests that model F1-score is above minimum threshold"""
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    
    f1 = f1_score(y_test, predictions)
    
    # Model should perform better than random
    min_f1 = 0.1
    assert f1 > min_f1, \
        f"F1-score ({f1:.4f}) is below minimum threshold ({min_f1})"

def test_model_has_probabilities(model, test_data):
    """Tests that model can predict probabilities"""
    X_test, _ = test_data
    
    # Check if model has predict_proba method
    assert hasattr(model, 'predict_proba'), \
        "Model doesn't have predict_proba method"
    
    probabilities = model.predict_proba(X_test)
    
    # Check shape
    assert probabilities.shape == (len(X_test), 2), \
        "Probability predictions have incorrect shape"
    
    # Check probabilities sum to 1
    prob_sums = probabilities.sum(axis=1)
    assert all(abs(prob_sums - 1.0) < 0.001), \
        "Probabilities don't sum to 1"

def test_model_recall_for_fraud(model, test_data):
    """Tests that model has reasonable recall for fraud detection"""
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    
    recall = recall_score(y_test, predictions)
    
    # For fraud detection, we want decent recall
    min_recall = 0.3
    assert recall > min_recall, \
        f"Recall ({recall:.4f}) is too low for fraud detection"

def test_model_feature_names(model):
    """Tests that model has correct feature names"""
    assert hasattr(model, 'feature_names_in_'), \
        "Model doesn't have feature_names_in_ attribute"
    
    feature_names = model.feature_names_in_
    assert len(feature_names) > 0, \
        "Model has no feature names"
    
    # Should have V1-V28 and Amount
    assert 'Amount' in feature_names, \
        "Model missing 'Amount' feature"
