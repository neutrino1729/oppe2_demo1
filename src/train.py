"""
Model Training Script with MLflow Tracking and Feast Integration
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os
import json
import subprocess
from feast import FeatureStore
from google.cloud import storage

def get_git_commit_hash():
    """Gets the current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def upload_to_gcs(bucket_name, source_path, dest_blob):
    """Uploads a file to GCS bucket."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_blob)
        blob.upload_from_filename(source_path)
        print(f"✅ Uploaded to gs://{bucket_name}/{dest_blob}")
    except Exception as e:
        print(f"⚠️  GCS upload skipped: {e}")

def train_model_with_feast():
    """
    Train fraud detection model using features from Feast
    """
    print("=" * 70)
    print("🚀 FRAUD DETECTION MODEL TRAINING WITH FEAST")
    print("=" * 70)
    
    # Initialize Feast store
    print("\n📡 Connecting to Feast feature store...")
    store = FeatureStore(repo_path="feature_repo")
    
    # Get feature names dynamically
    feature_view = store.get_feature_view(name="transaction_features")
    feature_names = [f"transaction_features:{f.name}" for f in feature_view.features]
    print(f"   Found {len(feature_names)} features")
    
    # Load entity data
    print("\n📊 Loading training data...")
    raw_data = pd.read_parquet("data/transactions.parquet")
    entity_df = raw_data[["transaction_id", "event_timestamp", "Class"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
    
    # Retrieve historical features
    print("   Retrieving features from Feast...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_names,
    ).to_df()
    
    print(f"   Training data shape: {training_df.shape}")
    
    # Set MLflow experiment
    mlflow.set_experiment("Fraud_Detection_Training")
    
    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"\n🔬 MLflow Run ID: {run.info.run_id}")
        
        # Prepare features and target
        X = training_df.drop(columns=["transaction_id", "event_timestamp", "Class", "location"])
        y = training_df["Class"]
        
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total samples: {len(X):,}")
        print(f"   Fraud cases: {y.sum():,} ({y.mean():.2%})")
        print(f"   Normal cases: {(~y.astype(bool)).sum():,}")
        
        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model parameters
        params = {
            'class_weight': 'balanced',
            'max_depth': 5,
            'min_samples_split': 100,
            'min_samples_leaf': 50,
            'random_state': 42
        }
        
        print(f"\n🔧 Training model with parameters:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Train model
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\n📊 Model Performance:")
        print(f"   F1-Score:    {f1:.4f}")
        print(f"   Precision:   {precision:.4f}")
        print(f"   Recall:      {recall:.4f}")
        print(f"   AUC-ROC:     {auc:.4f}")
        
        # Log parameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("git_commit", get_git_commit_hash())
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("feature_count", len(X.columns))
        
        # Log metrics to MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("fraud_rate", y.mean())
        
        # Log model to MLflow (only if not in CI)
        if not os.getenv('CI'):
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="fraud-detection-feast-dt"
            )
            print(f"\n✅ Model logged to MLflow")
        
        # Save model artifact locally
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save parameters to JSON
        params_path = "artifacts/params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"💾 Parameters saved to: {params_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    if os.getenv('CI'):
        # In CI, use local tracking
        mlflow.set_tracking_uri("file:./mlruns")
        print("🏃 Running in CI mode - logging to local mlruns")
    else:
        # Use remote server
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"🌐 MLflow tracking URI: {mlflow_uri}")
    
    train_model_with_feast()
