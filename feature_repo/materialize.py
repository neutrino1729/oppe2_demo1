"""
Materialize features to the online store
"""

import pandas as pd
from feast import FeatureStore

def materialize_features():
    print("🔄 Starting feature materialization...")
    
    store = FeatureStore(repo_path=".")
    
    # Load data to get timestamp range
    data_path = "../data/transactions.parquet"
    df = pd.read_parquet(data_path)
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    
    end_time = df['event_timestamp'].max()
    print(f"📅 Materializing up to {end_time}")
    
    store.materialize_incremental(end_date=end_time)
    print("✅ Materialization complete!")

if __name__ == "__main__":
    materialize_features()
