"""
Prepare data for Feast feature store
Converts CSV to Parquet and adds required columns
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path

def create_parquet_for_feast(input_csv_path: str, output_parquet_path: str):
    """
    Reads a raw transaction CSV, adds required columns for Feast,
    and saves it as a Parquet file.
    
    Args:
        input_csv_path: Path to the raw input CSV file
        output_parquet_path: Path to save the output Parquet file
    """
    print(f"🔄 Preparing {input_csv_path} for Feast...")
    
    try:
        df = pd.read_csv(input_csv_path)
        print(f"   Loaded {len(df):,} transactions")
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {input_csv_path}")
        return
    
    # Add a unique ID for each transaction
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = df.index
        print("   ✅ Added 'transaction_id' column")
    
    # Convert the 'Time' column to datetime
    if 'event_timestamp' not in df.columns:
        base_date = datetime(2022, 1, 1)
        df['event_timestamp'] = df['Time'].apply(
            lambda sec: base_date + timedelta(seconds=int(sec))
        )
        print("   ✅ Converted 'Time' to 'event_timestamp'")
    
    # Ensure output directory exists
    output_dir = Path(output_parquet_path).parent
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Parquet
    df.to_parquet(output_parquet_path, index=False)
    
    print(f"💾 Saved Feast-ready data to: {output_parquet_path}")
    print(f"   Columns: {list(df.columns)[:5]}... (+ {len(df.columns)-5} more)")
    print(f"   Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")
    print("-" * 60)

if __name__ == "__main__":
    # Prepare the main dataset
    input_path = "data/transactions.csv"
    output_path = "data/transactions.parquet"
    
    create_parquet_for_feast(input_path, output_path)
    print("✅ Data preparation for Feast complete!")
