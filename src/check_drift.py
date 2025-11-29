"""
Data Drift Detection using Evidently
Compares v0 (2022) vs v1 (2023) data distributions
"""

import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def check_data_drift(
    reference_path="data_orig/transactions_2022.csv",
    current_path="data_orig/transactions_2023.csv",
    output_dir="artifacts"
):
    """
    Compares reference (v0) and current (v1) datasets for drift
    
    Args:
        reference_path: Path to reference dataset (2022)
        current_path: Path to current dataset (2023)
        output_dir: Directory to save drift report
    """
    print("=" * 70)
    print("🌊 DATA DRIFT DETECTION")
    print("=" * 70)
    
    # Load datasets
    print(f"\n📊 Loading datasets...")
    try:
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)
        print(f"   ✅ Reference data (2022): {len(reference_df):,} transactions")
        print(f"   ✅ Current data (2023): {len(current_df):,} transactions")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please ensure data has been prepared with src/prepare_data.py")
        return
    
    # Show basic statistics
    print(f"\n📈 Dataset Comparison:")
    print(f"   Reference fraud rate: {reference_df['Class'].mean():.4%}")
    print(f"   Current fraud rate: {current_df['Class'].mean():.4%}")
    print(f"   Fraud rate change: {(current_df['Class'].mean() - reference_df['Class'].mean()):.4%}")
    
    # Create drift report
    print(f"\n🔍 Running Evidently drift analysis...")
    
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    # Run the report
    data_drift_report.run(
        reference_data=reference_df,
        current_data=current_df
    )
    
    # Save HTML report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "drift_report.html")
    data_drift_report.save_html(report_path)
    
    print(f"\n💾 Drift report saved to: {report_path}")
    
    # Extract drift metrics
    print(f"\n📊 Drift Summary:")
    print("-" * 70)
    
    # Get drift results
    drift_results = data_drift_report.as_dict()
    
    # Try to extract key metrics
    try:
        metrics = drift_results.get('metrics', [])
        dataset_drift_metric = None
        
        for metric in metrics:
            if metric.get('metric') == 'DatasetDriftMetric':
                dataset_drift_metric = metric.get('result', {})
                break
        
        if dataset_drift_metric:
            drift_share = dataset_drift_metric.get('drift_share', 0)
            num_drifted = dataset_drift_metric.get('number_of_drifted_columns', 0)
            num_columns = dataset_drift_metric.get('number_of_columns', 0)
            
            print(f"   • Total features analyzed: {num_columns}")
            print(f"   • Features with drift: {num_drifted}")
            print(f"   • Drift percentage: {drift_share * 100:.1f}%")
            
            if drift_share < 0.1:
                print(f"\n   ✅ LOW DRIFT: Data distributions are similar")
            elif drift_share < 0.3:
                print(f"\n   ⚠️  MODERATE DRIFT: Some distribution changes detected")
            else:
                print(f"\n   ❌ HIGH DRIFT: Significant distribution changes detected")
        else:
            print("   ℹ️  Detailed metrics available in HTML report")
            
    except Exception as e:
        print(f"   ⚠️  Could not extract summary metrics: {e}")
        print(f"   ℹ️  Full details available in HTML report")
    
    print("\n" + "=" * 70)
    print("✅ DRIFT DETECTION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 View detailed report: {report_path}")
    print("   (Download and open in browser)")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_data_drift()
