"""
IoT Anomaly Detection System
Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)
Models: Isolation Forest, Local Outlier Factor, One-Class SVM
Author: Mannat Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score, accuracy_score)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/ayman/Desktop/Mannat's Projects/IoT-Anomaly-Detection"
DATA_PATH = os.path.join(BASE_DIR, "data", "sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("IoT Anomaly Detection — AI4I 2020 Predictive Maintenance")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nClass distribution:")
print(df['Machine failure'].value_counts())
print(f"\nFailure rate: {df['Machine failure'].mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Select sensor features relevant to anomaly detection
feature_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Derived features (domain knowledge)
df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
df['wear_rate'] = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1)

feature_cols_extended = feature_cols + ['temp_diff', 'power', 'wear_rate']

X = df[feature_cols_extended].values
y = df['Machine failure'].values  # Ground truth for evaluation

print(f"\n✅ Features engineered: {len(feature_cols_extended)} features")
print(f"   Original: {feature_cols}")
print(f"   Derived:  ['temp_diff', 'power', 'wear_rate']")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert labels: 0 = normal(+1 for sklearn), 1 = anomaly(-1 for sklearn)
y_sklearn = np.where(y == 0, 1, -1)

print(f"\n✅ Data scaled with StandardScaler")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
# Contamination = known failure rate
contamination = df['Machine failure'].mean()
print(f"\n✅ Contamination rate: {contamination:.4f} ({contamination*100:.2f}%)")

# Model 1: Isolation Forest
print("\n" + "─" * 40)
print("Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)
iso_pred = iso_forest.fit_predict(X_scaled)
iso_binary = np.where(iso_pred == -1, 1, 0)

iso_f1 = f1_score(y, iso_binary)
iso_precision = precision_score(y, iso_binary)
iso_recall = recall_score(y, iso_binary)
iso_acc = accuracy_score(y, iso_binary)

print(f"  Accuracy:  {iso_acc*100:.2f}%")
print(f"  Precision: {iso_precision*100:.2f}%")
print(f"  Recall:    {iso_recall*100:.2f}%")
print(f"  F1-Score:  {iso_f1*100:.2f}%")

# Model 2: Local Outlier Factor
print("\n" + "─" * 40)
print("Training Local Outlier Factor...")
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    novelty=False
)
lof_pred = lof.fit_predict(X_scaled)
lof_binary = np.where(lof_pred == -1, 1, 0)

lof_f1 = f1_score(y, lof_binary)
lof_precision = precision_score(y, lof_binary)
lof_recall = recall_score(y, lof_binary)
lof_acc = accuracy_score(y, lof_binary)

print(f"  Accuracy:  {lof_acc*100:.2f}%")
print(f"  Precision: {lof_precision*100:.2f}%")
print(f"  Recall:    {lof_recall*100:.2f}%")
print(f"  F1-Score:  {lof_f1*100:.2f}%")

# Model 3: One-Class SVM (on subset for speed)
print("\n" + "─" * 40)
print("Training One-Class SVM (on 3000 sample subset for speed)...")
svm_idx = np.random.RandomState(42).choice(len(X_scaled), 3000, replace=False)
X_svm = X_scaled[svm_idx]
y_svm = y[svm_idx]

ocsvm = OneClassSVM(kernel='rbf', nu=contamination, gamma='auto')
svm_pred = ocsvm.fit_predict(X_svm)
svm_binary = np.where(svm_pred == -1, 1, 0)

svm_f1 = f1_score(y_svm, svm_binary)
svm_precision = precision_score(y_svm, svm_binary)
svm_recall = recall_score(y_svm, svm_binary)
svm_acc = accuracy_score(y_svm, svm_binary)

print(f"  Accuracy:  {svm_acc*100:.2f}%")
print(f"  Precision: {svm_precision*100:.2f}%")
print(f"  Recall:    {svm_recall*100:.2f}%")
print(f"  F1-Score:  {svm_f1*100:.2f}%")

# ─────────────────────────────────────────────
# 5. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
results = {
    'Isolation Forest': {'Accuracy': iso_acc, 'Precision': iso_precision,
                          'Recall': iso_recall, 'F1': iso_f1},
    'Local Outlier Factor': {'Accuracy': lof_acc, 'Precision': lof_precision,
                              'Recall': lof_recall, 'F1': lof_f1},
    'One-Class SVM': {'Accuracy': svm_acc, 'Precision': svm_precision,
                       'Recall': svm_recall, 'F1': svm_f1},
}

for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value*100:.2f}%")

# Best model
best_model_name = max(results, key=lambda m: results[m]['F1'])
print(f"\n🏆 Best Model: {best_model_name} (F1: {results[best_model_name]['F1']*100:.2f}%)")

# ─────────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────────
joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

# Save metrics for README
with open(os.path.join(MODELS_DIR, 'metrics.txt'), 'w') as f:
    f.write(f"Best Model: Isolation Forest\n")
    f.write(f"Accuracy:   {iso_acc*100:.2f}%\n")
    f.write(f"Precision:  {iso_precision*100:.2f}%\n")
    f.write(f"Recall:     {iso_recall*100:.2f}%\n")
    f.write(f"F1-Score:   {iso_f1*100:.2f}%\n")
    f.write(f"Dataset:    AI4I 2020 Predictive Maintenance (UCI ML Repository)\n")
    f.write(f"Samples:    {len(df)}\n")
    f.write(f"Features:   {len(feature_cols_extended)}\n")

print(f"✅ Model saved to {MODELS_DIR}/isolation_forest.pkl")
print(f"✅ Scaler saved to {MODELS_DIR}/scaler.pkl")
print(f"✅ Metrics saved to {MODELS_DIR}/metrics.txt")

# ─────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IoT Anomaly Detection — AI4I 2020 Dataset', fontsize=14, fontweight='bold')

# Plot 1: Anomaly Score Distribution
scores = iso_forest.decision_function(X_scaled)
axes[0, 0].hist(scores[y == 0], bins=50, alpha=0.7, label='Normal', color='steelblue')
axes[0, 0].hist(scores[y == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
axes[0, 0].set_title('Anomaly Score Distribution')
axes[0, 0].set_xlabel('Isolation Forest Score')
axes[0, 0].legend()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y, iso_binary)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
axes[0, 1].set_title('Confusion Matrix — Isolation Forest')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')

# Plot 3: Feature Correlation
corr = pd.DataFrame(X_scaled, columns=feature_cols_extended).corr()
sns.heatmap(corr, ax=axes[1, 0], cmap='coolwarm', center=0, annot=False)
axes[1, 0].set_title('Sensor Feature Correlation')

# Plot 4: Model Comparison Bar Chart
model_names = list(results.keys())
f1_scores = [results[m]['F1'] * 100 for m in model_names]
colors = ['#2196F3', '#FF9800', '#4CAF50']
axes[1, 1].bar(model_names, f1_scores, color=colors, edgecolor='black', linewidth=0.5)
axes[1, 1].set_title('F1-Score Comparison Across Models')
axes[1, 1].set_ylabel('F1-Score (%)')
axes[1, 1].set_ylim(0, 100)
for i, v in enumerate(f1_scores):
    axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'evaluation_report.png'), dpi=150, bbox_inches='tight')
print("✅ Evaluation charts saved to ../models/evaluation_report.png")
plt.show()

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
