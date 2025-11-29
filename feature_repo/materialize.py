"""
Materialize features to the online store
Note: This is skipped in CI environments
"""

import pandas as pd
from feast import FeatureStore
import os

def materialize_features():
    # Skip in CI
    if os.getenv('CI'):
        print("ℹ️  Skipping materialization in CI environment")
        return
    
    print("🔄 Starting feature materialization...")
    
    store = FeatureStore(repo_path=".")
    
    # Load data to get timestamp range
    data_path = "../data/transactions.parquet"
    
    if not os.path.exists(data_path):
        print(f"⚠️  Data file not found: {data_path}")
        print("   Run data preparation scripts first")
        return
    
    df = pd.read_parquet(data_path)
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    
    end_time = df['event_timestamp'].max()
    print(f"📅 Materializing up to {end_time}")
    
    store.materialize_incremental(end_date=end_time)
    print("✅ Materialization complete!")

if __name__ == "__main__":
    materialize_features()
