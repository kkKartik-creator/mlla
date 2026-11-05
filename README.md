# AI-Based Network Intrusion Detection System Using Machine Learning and Hyperparameter Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

An end-to-end Machine Learning-powered **Network Intrusion Detection System (NIDS)** developed on the benchmark **UNSW-NB15** dataset. The system analyzes network flow traffic, differentiates legitimate network flows from cyber intrusions, identifies specific granular attack categories, optimizes classification decision thresholds, and provides an interactive Security Operations Center (SOC) dashboard.

---

## 📌 Project Overview & Highlights

- **Dual-Tier Classification Engine**:
  - **Tier 1 (Binary)**: Differentiates **Normal (0)** traffic from **Malicious Attack (1)** traffic.
  - **Tier 2 (Multiclass)**: Classifies attack vectors into specific attack families (**Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms**).
- **Five Machine Learning Algorithms Implemented & Compared**:
  1. *Logistic Regression*
  2. *Decision Tree Classifier*
  3. *Random Forest Classifier*
  4. *Support Vector Machine (SVM)*
  5. *XGBoost Classifier*
- **Hyperparameter Optimization**: Systematic parameter exploration and tuning using `RandomizedSearchCV` and `GridSearchCV` with **Stratified 5-Fold Cross-Validation** (optimizing for F1-Score).
- **Classification Threshold Calibration**: Validation-based threshold tuning to balance high attack recall against false alarms (minimizing SOC alert fatigue).
- **Strict Anti-Data-Leakage Design**: Transformers and scalers fitted strictly on training data; evaluation conducted on an untouched held-out test split.
- **Interactive Security Monitoring Dashboard**: Streamlit dashboard featuring live telemetry inspection, CSV packet batch prediction, threat severity scoring, and alert management.

---

## 📂 Project Directory Structure

```text
d:\mlproject\
│
├── Network_Intrusion_Detection.ipynb   # Complete practical Jupyter Notebook (Sections 1 to 12)
├── app.py                             # Streamlit Security Monitoring Dashboard (Section 13)
├── requirements.txt                   # Project dependencies and library versions
├── README.md                          # Project documentation and execution instructions
│
├── best_model.pkl                     # Serialized best tuned binary model + multiclass classifier bundle
├── preprocessing.pkl                  # Serialized ColumnTransformer (StandardScaler + OneHotEncoder)
├── feature_columns.pkl                # Metadata of numerical, categorical, and transformed features
├── model_metadata.pkl                 # Evaluation metrics, confusion matrix, ROC AUC, tuning history
│
├── data/
│   ├── UNSW_NB15.csv                  # Full UNSW-NB15 network flow dataset
│   └── sample_test_traffic.csv        # Pre-packaged sample test batch for dashboard instant prediction
│
└── figures/                           # Saved high-resolution evaluation and EDA charts
    ├── eda_class_distribution.png
    ├── eda_attack_categories.png
    ├── eda_feature_correlation.png
    ├── eda_numerical_distributions.png
    ├── split_ratio_comparison.png
    ├── default_models_comparison.png
    ├── tuning_comparison_all_models.png
    ├── tuning_comparison_individual.png
    ├── threshold_optimization.png
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites & Environment Setup

Ensure you have **Python 3.10+** installed. In your terminal (PowerShell, Command Prompt, or Bash), navigate to the project directory:

```bash
cd d:\mlproject
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. How to Open and Run the Jupyter Notebook

The file **`Network_Intrusion_Detection.ipynb`** is organized into 12 structured, sequential practical sections with code cells, markdown explanations, and intermediate output displays.

#### Option A: Running in VS Code / Cursor
1. Open the folder `d:\mlproject` in VS Code.
2. Click on `Network_Intrusion_Detection.ipynb` in the file explorer.
3. Select your Python 3 kernel in the top-right corner.
4. Click **Run All** (or run cells sequentially using `Shift + Enter`).

#### Option B: Running in Jupyter Lab / Notebook
Launch Jupyter Notebook from your terminal:

```bash
jupyter notebook Network_Intrusion_Detection.ipynb
```
Select **Kernel -> Restart & Run All** to execute all cells from start to finish.

---

### 3. How to Launch the Streamlit Security Dashboard

Launch the interactive web application by executing:

```bash
streamlit run app.py
```

The application will start and open automatically in your browser at:
`http://localhost:8501`

#### Dashboard Capabilities:
- **📊 System Overview**: Displays total records analyzed, normal flows, detected attacks, attack percentage, and SOC health metrics.
- **🔍 Attack Analysis**: Interactive breakdowns of specific attack categories, volume distributions, severity rankings, and EDA plots.
- **⚡ Model Performance & Tuning**: Comparative tables comparing Accuracy, Precision, Recall, F1-Score, and FPR before vs. after tuning across all 5 models, plus ROC curve and confusion matrix.
- **🎯 Traffic Prediction Engine**: Upload any network flow CSV file or click **"📁 Load Pre-Packaged Test Sample"** to instantly run real-time inference with probability scores, binary classifications, attack category tags, and downloadable CSV reports.
- **🚨 Security Alert Center**: Live incident queue with severity badges (**CRITICAL**, **HIGH**, **MEDIUM**), flow telemetry, and automated defensive mitigation recommendations.

---

## 📊 Experimental Results & Model Comparison

### Performance Comparison: Default vs. Tuned Models (Held-Out Test Set)

| Algorithm | Stage | Accuracy | Precision | Recall | F1-Score | False Positive Rate (FPR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **Tuned** | **94.88%** | **96.42%** | **96.11%** | **0.9626** | **8.18%** |
| Random Forest | Tuned | 94.75% | 96.19% | 96.15% | 0.9617 | 8.75% |
| Decision Tree | Tuned | 93.87% | 95.34% | 95.69% | 0.9551 | 10.74% |
| Support Vector Machine | Tuned | 93.38% | 94.88% | 95.45% | 0.9516 | 11.83% |
| Logistic Regression | Tuned | 93.38% | 94.88% | 95.45% | 0.9516 | 11.83% |

> **Best Overall Performer**: **XGBoost Classifier** achieved the highest overall **F1-Score (0.9626)** and lowest **False Positive Rate (8.18%)**, making it the primary choice for the production binary detection engine.

---

## 🛡️ Cybersecurity Domain Significance: FP vs. FN

In real-world Security Operations Centers (SOC):
- **False Positives (FP - Type I Error)**:
  - *Definition*: Benign business traffic erroneously flagged as an intrusion.
  - *Impact*: Leads to severe **alert fatigue** for security analysts, potential blocking of critical business communications, and degraded service availability.
- **False Negatives (FN - Type II Error)**:
  - *Definition*: Malicious cyber attacks missed by the detection system.
  - *Impact*: Results in **undetected breaches**, persistence, lateral movement, ransomware infection, or confidential data exfiltration.

By tuning the **probability classification threshold** (calibrated to $\approx 0.40$), the system ensures high attack capture (**Recall > 96%**) while keeping false alarms strictly bounded.

---

## 📦 Required Libraries Summary

- `numpy` & `pandas`: Data structures and vector math
- `matplotlib` & `seaborn`: Static charts and confusion matrix heatmaps
- `scikit-learn`: Preprocessing, pipelines, baseline models, cross-validation, and metrics
- `xgboost`: Extreme Gradient Boosting ensemble classifier
- `joblib`: Model serialization and persistence
- `streamlit` & `plotly`: Interactive security monitoring web dashboard

---

## 📜 Academic Attribution & License
- **Dataset**: UNSW-NB15 created by the Cyber Range Lab of UNSW Canberra.
- **License**: Educational and research use.

