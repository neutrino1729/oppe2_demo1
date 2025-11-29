"""
Generate SHAP explanations for model interpretability
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import sys

sys.stdout.reconfigure(line_buffering=True)

def generate_shap_explanations():
    """
    Loads trained model and generates SHAP explanations
    """
    print("=" * 70)
    print("🧠 GENERATING MODEL EXPLANATIONS (SHAP)")
    print("=" * 70)
    
    try:
        # Load model
        model = joblib.load("artifacts/model.pkl")
        print("✅ Model loaded successfully")
        
        # Load data
        df = pd.read_csv("data/transactions.csv")
        print(f"✅ Data loaded: {len(df):,} transactions")
        
        # Get expected features
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y = df['Class']
        
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        return
    
    # Create test set
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Use smaller sample for performance
    sample_size = min(2000, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    
    print(f"\n📊 Calculating SHAP values for {len(X_test_sample)} samples...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_sample)
    
    # Create output directory
    os.makedirs("artifacts", exist_ok=True)
    
    # 1. Global Summary Plot (Bar)
    print("\n📈 Creating global feature importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: artifacts/shap_summary.png")
    
    # 2. Detailed Summary Plot (Beeswarm)
    print("\n📈 Creating detailed SHAP beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Feature Impact", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("artifacts/shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: artifacts/shap_beeswarm.png")
    
    # 3. Force Plot (HTML)
    print("\n🌐 Creating interactive force plot...")
    force_plot = shap.force_plot(
        base_value=explainer.expected_value[1],
        shap_values=shap_values.values[:, :, 1],
        features=X_test_sample,
        matplotlib=False
    )
    shap.save_html("artifacts/shap_force_plot_all.html", force_plot)
    print("   ✅ Saved: artifacts/shap_force_plot_all.html")
    
    # 4. Generate Text Report
    print("\n📝 Creating text report...")
    
    # Calculate feature importance
    if hasattr(model, "classes_") and 1 in model.classes_:
        positive_class_index = np.where(model.classes_ == 1)[0][0]
        shap_values_for_importance = shap_values.values[:, :, positive_class_index]
    else:
        shap_values_for_importance = shap_values.values
    
    importance_df = pd.DataFrame({
        'feature': X_test_sample.columns,
        'importance': np.abs(shap_values_for_importance).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    report_path = 'artifacts/shap_report.txt'
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("🧠 MODEL EXPLAINABILITY RESULTS (SHAP)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Top 10 Most Important Features:\n")
        f.write("-" * 70 + "\n")
        f.write(importance_df.head(10).to_string(index=False))
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("📋 Key Insights:\n")
        f.write("-" * 70 + "\n")
        top_feature = importance_df.iloc[0]
        f.write(f"• Most predictive feature: {top_feature['feature']}\n")
        f.write(f"  Average SHAP value: {top_feature['importance']:.4f}\n\n")
        f.write("• Feature importance indicates which transaction patterns\n")
        f.write("  are most suspicious for fraud detection.\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   ✅ Saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ SHAP EXPLANATION GENERATION COMPLETE!")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print("   • artifacts/shap_summary.png")
    print("   • artifacts/shap_beeswarm.png")
    print("   • artifacts/shap_force_plot_all.html")
    print("   • artifacts/shap_report.txt")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    generate_shap_explanations()
