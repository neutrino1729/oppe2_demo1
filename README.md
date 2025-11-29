# 🚀 MLOps Fraud Detection System

[![MLOps Pipeline](https://github.com/neutrino1729/oppe2_demo1/actions/workflows/mlops-pipeline.yaml/badge.svg)](https://github.com/neutrino1729/oppe2_demo1/actions/workflows/mlops-pipeline.yaml)

A production-grade fraud detection system with complete MLOps pipeline including:

## 🎯 Features

- ✅ **Real-time Fraud Detection** using Decision Tree classifier
- ✅ **Feature Store** with Feast for consistent feature management
- ✅ **Experiment Tracking** with MLflow
- ✅ **Data Versioning** with DVC
- ✅ **Model Explainability** using SHAP
- ✅ **Fairness Auditing** with Fairlearn
- ✅ **Data Drift Detection** for model monitoring
- ✅ **Poisoning Detection** for data security
- ✅ **Automated Testing** with pytest
- ✅ **CI/CD Pipeline** with GitHub Actions

## 📊 Dataset

Credit card transactions dataset with:
- 284,807 transactions
- 31 features (anonymized PCA components)
- Highly imbalanced (~0.17% fraud rate)

## 🏗️ Architecture
```
Data (GCS) → DVC → Feast Feature Store → Model Training → MLflow
                                              ↓
                                         Validation
                                              ↓
                                    GitHub Actions CI/CD
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud account
- GitHub account

### Installation
```bash
# Clone repository
git clone https://github.com/neutrino1729/oppe2_demo1.git
cd oppe2_demo1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure DVC
dvc pull
```

### Training
```bash
# Apply Feast features
cd feature_repo
feast apply
python materialize.py
cd ..

# Train model
python src/train.py
```

### Validation
```bash
# Run all tests
pytest tests/ -v

# Check explainability
python src/generate_explanations.py

# Check fairness
python src/check_fairness.py

# Check drift
python src/check_drift_simple.py

# Check poisoning
python src/check_poisoning.py
```

## 📁 Project Structure
```
oppe2_demo1/
├── .github/workflows/       # CI/CD pipelines
├── src/                     # Source code
│   ├── train.py            # Model training
│   ├── generate_explanations.py  # SHAP
│   ├── check_fairness.py   # Fairness audit
│   ├── check_drift_simple.py     # Drift detection
│   └── check_poisoning.py  # Poisoning detection
├── tests/                   # Automated tests
├── feature_repo/           # Feast feature definitions
├── data/                   # Training data (DVC tracked)
└── artifacts/              # Model outputs
```

## 🔒 Responsible AI

This project implements comprehensive responsible AI practices:

- **Explainability**: SHAP values for model interpretability
- **Fairness**: Demographic parity auditing
- **Monitoring**: Drift detection between data versions
- **Security**: Poisoning attack detection
- **Testing**: Automated quality gates

## 📈 Results

- **F1-Score**: 0.15+
- **Recall**: 0.87 (catches 87% of fraud)
- **AUC-ROC**: 0.92+

## 🤝 Contributing

This is a learning project for MLOps best practices.

## 📝 License

Educational project - MIT License
