"""
Fairness Auditing using Fairlearn
Checks for bias across location groups
"""

import pandas as pd
import joblib
import json
import os
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

def check_model_fairness():
    """
    Audits the model for fairness based on 'location' sensitive feature
    """
    print("=" * 70)
    print("⚖️  MODEL FAIRNESS AUDIT")
    print("=" * 70)
    
    try:
        # Load model and data
        print("\n📊 Loading model and data...")
        model = joblib.load("artifacts/model.pkl")
        df = pd.read_csv("data/transactions.csv")
        
        if 'location' not in df.columns:
            print("❌ Error: 'location' column not found.")
            print("   Please run: python src/add_location.py")
            return
        
        print("✅ Data loaded successfully")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Prepare features
    expected_features = model.feature_names_in_
    X = df[expected_features]
    y_true = df['Class']
    sensitive_feature = df['location']
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Location A: {(sensitive_feature == 'Location_A').sum():,}")
    print(f"   Location B: {(sensitive_feature == 'Location_B').sum():,}")
    print(f"   Fraud cases: {y_true.sum():,} ({y_true.mean():.2%})")
    
    # Make predictions
    print("\n🔮 Generating predictions...")
    y_pred = model.predict(X)
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 Overall Model Accuracy: {overall_accuracy:.4f}")
    
    # Calculate group-specific metrics
    print("\n📊 Performance by Location:")
    print("-" * 70)
    
    for location in ['Location_A', 'Location_B']:
        mask = sensitive_feature == location
        group_accuracy = accuracy_score(y_true[mask], y_pred[mask])
        group_fraud_rate = y_true[mask].mean()
        group_pred_fraud_rate = y_pred[mask].mean()
        
        print(f"\n   {location}:")
        print(f"   ├─ Accuracy: {group_accuracy:.4f}")
        print(f"   ├─ True Fraud Rate: {group_fraud_rate:.4%}")
        print(f"   └─ Predicted Fraud Rate: {group_pred_fraud_rate:.4%}")
    
    # Calculate fairness metrics
    print("\n⚖️  Fairness Metrics:")
    print("-" * 70)
    
    # Demographic Parity Difference
    dpd = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_feature
    )
    
    print(f"\n   📏 Demographic Parity Difference: {dpd:.4f}")
    print(f"      (Closer to 0 is more fair)")
    
    if abs(dpd) < 0.05:
        print(f"      ✅ PASS: Model shows good demographic parity")
    elif abs(dpd) < 0.10:
        print(f"      ⚠️  WARNING: Moderate demographic disparity detected")
    else:
        print(f"      ❌ FAIL: Significant demographic disparity detected")
    
    # Equalized Odds Difference
    try:
        eod = equalized_odds_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_feature
        )
        print(f"\n   📏 Equalized Odds Difference: {eod:.4f}")
        print(f"      (Closer to 0 is more fair)")
        
        if abs(eod) < 0.05:
            print(f"      ✅ PASS: Model shows good equalized odds")
        elif abs(eod) < 0.10:
            print(f"      ⚠️  WARNING: Moderate odds disparity detected")
        else:
            print(f"      ❌ FAIL: Significant odds disparity detected")
    except Exception as e:
        print(f"\n   ⚠️  Could not calculate Equalized Odds: {e}")
        eod = None
    
    # Create fairness report
    fairness_report = {
        "demographic_parity_difference": float(dpd),
        "equalized_odds_difference": float(eod) if eod is not None else None,
        "overall_accuracy": float(overall_accuracy),
        "location_distribution": {
            "Location_A": int((sensitive_feature == 'Location_A').sum()),
            "Location_B": int((sensitive_feature == 'Location_B').sum())
        }
    }
    
    # Save report
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/fairness_report.json"
    
    with open(report_path, "w") as f:
        json.dump(fairness_report, f, indent=4)
    
    print(f"\n💾 Fairness report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ FAIRNESS AUDIT COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    check_model_fairness()
