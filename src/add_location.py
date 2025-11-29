"""
Add synthetic 'location' feature for fairness analysis
"""

import pandas as pd
import numpy as np
import os

def add_sensitive_feature(data_path="data/transactions.csv"):
    """
    Adds a synthetic 'location' column to the dataset for fairness analysis.
    
    Args:
        data_path (str): Path to the transaction data CSV.
    """
    print(f"📍 Adding sensitive feature to {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    if 'location' in df.columns:
        print("⚠️  'location' column already exists. Skipping.")
        return
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Add random location (50/50 split)
    df['location'] = np.random.choice(['Location_A', 'Location_B'], size=len(df))
    
    # Save back to file
    df.to_csv(data_path, index=False)
    
    print(f"✅ Successfully added 'location' column")
    print(f"   Location_A: {(df['location'] == 'Location_A').sum():,} transactions")
    print(f"   Location_B: {(df['location'] == 'Location_B').sum():,} transactions")
    print("-" * 60)

if __name__ == "__main__":
    add_sensitive_feature()
