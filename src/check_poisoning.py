"""
Data Poisoning Detection using KNN
Identifies potentially flipped labels (suspicious data points)
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse
import os

def find_suspicious_labels(data_path, k=5, threshold=0.5):
    """
    Analyzes dataset to find rows with potentially flipped labels using KNN
    
    Args:
        data_path (str): Path to the CSV data file
        k (int): Number of neighbors to consider
        threshold (float): Fraction of neighbors that must disagree to flag a point
    
    Returns:
        list: Indices of suspicious data points
    """
    print("=" * 70)
    print("🛡️  DATA POISONING DETECTION")
    print("=" * 70)
    
    print(f"\n📊 Analyzing: {data_path}")
    print(f"   K neighbors: {k}")
    print(f"   Threshold: {threshold}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"\n✅ Data loaded: {len(df):,} transactions")
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}")
        return []
    
    # Prepare features (drop non-feature columns)
    # Handle both cases: with and without 'location' column
    columns_to_drop = ['Class', 'Time']
    if 'location' in df.columns:
        columns_to_drop.append('location')
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df['Class']
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Fraud cases: {y.sum():,} ({y.mean():.2%})")
    print(f"   Normal cases: {(~y.astype(bool)).sum():,}")
    
    # Use KNN to find neighbors
    print(f"\n🔍 Running KNN analysis...")
    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(X, y)
    
    # Find k+1 nearest neighbors for every point (including itself)
    distances, indices = knn.kneighbors(X)
    
    suspicious_indices = []
    suspicious_details = []
    
    # Check each data point
    for i in range(len(df)):
        original_label = y.iloc[i]
        neighbor_indices = indices[i][1:]  # Exclude the point itself
        neighbor_labels = y.iloc[neighbor_indices]
        
        # Count how many neighbors have different label
        num_mismatched = np.sum(neighbor_labels != original_label)
        mismatch_ratio = num_mismatched / k
        
        # If mismatch ratio exceeds threshold, flag as suspicious
        if mismatch_ratio >= threshold:
            suspicious_indices.append(i)
            suspicious_details.append({
                'index': i,
                'label': original_label,
                'mismatch_ratio': mismatch_ratio,
                'neighbor_labels': neighbor_labels.tolist()
            })
    
    # Print results
    print(f"\n📊 Detection Results:")
    print("-" * 70)
    print(f"   Total samples analyzed: {len(df):,}")
    print(f"   Suspicious labels found: {len(suspicious_indices):,}")
    print(f"   Suspicion rate: {len(suspicious_indices) / len(df):.2%}")
    
    if len(suspicious_indices) > 0:
        print(f"\n   First 10 suspicious indices: {suspicious_indices[:10]}")
        
        # Show breakdown by original label
        suspicious_fraud = sum(1 for i in suspicious_indices if y.iloc[i] == 1)
        suspicious_normal = len(suspicious_indices) - suspicious_fraud
        
        print(f"\n   Breakdown:")
        print(f"      • Suspicious fraud labels: {suspicious_fraud}")
        print(f"      • Suspicious normal labels: {suspicious_normal}")
    
    # Assessment
    print(f"\n🎯 Assessment:")
    print("-" * 70)
    
    suspicion_rate = len(suspicious_indices) / len(df)
    
    if suspicion_rate < 0.01:
        print(f"   ✅ CLEAN: Very low suspicion rate (<1%)")
        status = "PASS"
    elif suspicion_rate < 0.05:
        print(f"   ⚠️  MODERATE: Some suspicious labels detected (1-5%)")
        status = "WARNING"
    else:
        print(f"   ❌ HIGH RISK: Many suspicious labels (>5%)")
        status = "FAIL"
    
    # Save detailed report
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/poisoning_report.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("🛡️  DATA POISONING DETECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Total samples: {len(df):,}\n")
        f.write(f"Suspicious samples: {len(suspicious_indices):,}\n")
        f.write(f"Suspicion rate: {suspicion_rate:.2%}\n")
        f.write(f"Status: {status}\n\n")
        f.write("=" * 70 + "\n")
        f.write("Top 20 Suspicious Data Points:\n")
        f.write("=" * 70 + "\n")
        
        for detail in suspicious_details[:20]:
            f.write(f"\nIndex: {detail['index']}\n")
            f.write(f"  Original Label: {detail['label']}\n")
            f.write(f"  Mismatch Ratio: {detail['mismatch_ratio']:.2%}\n")
            f.write(f"  Neighbor Labels: {detail['neighbor_labels']}\n")
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ POISONING DETECTION COMPLETE!")
    print("=" * 70)
    
    return suspicious_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for suspicious (potentially poisoned) labels in a dataset"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/transactions.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to check"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fraction of neighbors that must disagree to flag a point"
    )
    
    args = parser.parse_args()
    
    find_suspicious_labels(
        data_path=args.data_path,
        k=args.k,
        threshold=args.threshold
    )
