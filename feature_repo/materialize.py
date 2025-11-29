"""
Materialize features to the online store
"""

import pandas as pd
from feast import FeatureStore
from datetime import datetime

def materialize_features():
    """
    Materializes features from offline to online store
    """
    print("🔄 Starting feature materialization...")
    
    # Load the feature store
    store = FeatureStore(repo_path=".")
    
    # Load data to get timestamp range
    data_path = "../data/transactions.parquet"
    
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    if df.empty or 'event_timestamp' not in df.columns:
        raise ValueError("❌ Data is empty or missing 'event_timestamp'")
    
    # Convert timestamp column
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    
    # Get time range
    start_time = df['event_timestamp'].min()
    end_time = df['event_timestamp'].max()
    
    print(f"📅 Materializing features from {start_time} to {end_time}...")
    
    # Materialize features
    store.materialize_incremental(end_date=end_time)
    
    print("✅ Feature materialization complete!")
    print("-" * 60)

if __name__ == "__main__":
    materialize_features()
