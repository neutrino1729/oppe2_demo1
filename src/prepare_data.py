"""
Data Preparation Script
Splits transactions.csv into v0 (2022) and v1 (2023) based on Time column
"""

import pandas as pd
import os

def split_transactions(input_path: str, output_dir: str):
    """
    Split transaction data into two time periods
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save split files
    """
    print(f"📖 Reading data from {input_path}...")
    
    # Read the CSV
    df = pd.read_csv(input_path)
    print(f"   Total transactions: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")
    
    # Sort by Time column
    df = df.sort_values("Time").reset_index(drop=True)
    
    # Split into two halves
    midpoint = len(df) // 2
    df_2022 = df.iloc[:midpoint].copy()
    df_2023 = df.iloc[midpoint:].copy()
    
    print(f"\n📊 Split Statistics:")
    print(f"   2022 data: {len(df_2022):,} transactions")
    print(f"   2023 data: {len(df_2023):,} transactions")
    print(f"   Fraud rate 2022: {df_2022['Class'].mean():.4%}")
    print(f"   Fraud rate 2023: {df_2023['Class'].mean():.4%}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the files
    path_2022 = os.path.join(output_dir, "transactions_2022.csv")
    path_2023 = os.path.join(output_dir, "transactions_2023.csv")
    
    df_2022.to_csv(path_2022, index=False)
    df_2023.to_csv(path_2023, index=False)
    
    print(f"\n💾 Saved files:")
    print(f"   ✅ {path_2022}")
    print(f"   ✅ {path_2023}")
    print(f"\n✅ Data preparation complete!")

if __name__ == "__main__":
    input_csv_path = "orig_data/transactions.csv"
    output_directory = "data_orig"
    
    split_transactions(input_csv_path, output_directory)
