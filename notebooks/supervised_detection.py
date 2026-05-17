"""
IoT Predictive Maintenance — Supervised ML Pipeline
Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)
Models: Random Forest, Isolation Forest (unsupervised baseline comparison)
Author: Mannat Singh
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/ayman/Desktop/Mannat's Projects/IoT-Anomaly-Detection"
DATA_PATH = os.path.join(BASE_DIR, "data", "sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("=" * 65)
print("IoT Predictive Maintenance — Supervised ML Pipeline")
print("Dataset: AI4I 2020 (UCI Machine Learning Repository)")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

failure_rate = df['Machine failure'].mean()
print(f"   Failure rate: {failure_rate*100:.2f}%  (Highly imbalanced dataset)")
print(f"   Normal samples:  {(df['Machine failure']==0).sum():,}")
print(f"   Failure samples: {(df['Machine failure']==1).sum():,}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
feature_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Domain-knowledge derived features
df['temp_diff']   = df['Process temperature [K]'] - df['Air temperature [K]']
df['power']       = df['Rotational speed [rpm]'] * df['Torque [Nm]']
df['wear_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['speed_torque_ratio'] = df['Rotational speed [rpm]'] / (df['Torque [Nm]'] + 1)

all_features = feature_cols + ['temp_diff', 'power', 'wear_torque', 'speed_torque_ratio']

X = df[all_features]
y = df['Machine failure']

print(f"\n✅ Features engineered: {len(all_features)} total")
print(f"   Raw sensor:   {feature_cols}")
print(f"   Derived:      ['temp_diff', 'power', 'wear_torque', 'speed_torque_ratio']")

# ─────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT & SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✅ Train/Test split (80/20, stratified):")
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. SUPERVISED — RANDOM FOREST (with class balancing)
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("Training Random Forest (class_weight='balanced')...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',  # Handles class imbalance
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_prob = rf.predict_proba(X_test_scaled)[:, 1]

rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec  = recall_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf_prob)

print(f"\n  Random Forest Results:")
print(f"  Accuracy:  {rf_acc*100:.2f}%")
print(f"  Precision: {rf_prec*100:.2f}%")
print(f"  Recall:    {rf_rec*100:.2f}%")
print(f"  F1-Score:  {rf_f1*100:.2f}%")
print(f"  ROC-AUC:   {rf_auc*100:.2f}%")

# Cross validation
cv_f1 = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"  5-Fold CV F1: {cv_f1.mean()*100:.2f}% (±{cv_f1.std()*100:.2f}%)")

# ─────────────────────────────────────────────
# 5. UNSUPERVISED — ISOLATION FOREST (comparison baseline)
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("Training Isolation Forest (unsupervised baseline)...")

iso = IsolationForest(n_estimators=200, contamination=failure_rate, random_state=42, n_jobs=-1)
iso.fit(X_train_scaled)
iso_pred_raw = iso.predict(X_test_scaled)
iso_pred = np.where(iso_pred_raw == -1, 1, 0)

iso_f1   = f1_score(y_test, iso_pred)
iso_prec = precision_score(y_test, iso_pred)
iso_rec  = recall_score(y_test, iso_pred)
iso_acc  = accuracy_score(y_test, iso_pred)

print(f"\n  Isolation Forest Results (no labels used):")
print(f"  Accuracy:  {iso_acc*100:.2f}%")
print(f"  F1-Score:  {iso_f1*100:.2f}%")

# ─────────────────────────────────────────────
# 6. FULL REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL EVALUATION REPORT")
print("=" * 65)
print(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>12} {'Recall':>10}")
print("-" * 70)
print(f"{'Random Forest'::<25} {rf_acc*100:>9.2f}% {rf_f1*100:>9.2f}% {rf_prec*100:>11.2f}% {rf_rec*100:>9.2f}%")
print(f"{'Isolation Forest'::<25} {iso_acc*100:>9.2f}% {iso_f1*100:>9.2f}% {iso_prec*100:>11.2f}% {iso_rec*100:>9.2f}%")

print(f"\n🏆 Best Model: Random Forest")
print(f"   ROC-AUC: {rf_auc*100:.2f}%")
print(f"   Cross-Validation F1: {cv_f1.mean()*100:.2f}% (±{cv_f1.std()*100:.2f}%)")

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=all_features).sort_values(ascending=False)
print(f"\nTop Feature Importances:")
for feat, imp in feat_imp.items():
    print(f"  {feat:<40} {imp:.4f}")

# ─────────────────────────────────────────────
# 8. SAVE MODELS & METRICS
# ─────────────────────────────────────────────
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

with open(os.path.join(MODELS_DIR, 'metrics.txt'), 'w') as f:
    f.write("=== IoT Predictive Maintenance — Model Evaluation ===\n\n")
    f.write("Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Features: {len(all_features)} (5 raw sensors + 4 engineered)\n")
    f.write(f"Train/Test Split: 80/20 (stratified)\n\n")
    f.write("--- Random Forest (Best Model) ---\n")
    f.write(f"Accuracy:            {rf_acc*100:.2f}%\n")
    f.write(f"Precision:           {rf_prec*100:.2f}%\n")
    f.write(f"Recall:              {rf_rec*100:.2f}%\n")
    f.write(f"F1-Score:            {rf_f1*100:.2f}%\n")
    f.write(f"ROC-AUC:             {rf_auc*100:.2f}%\n")
    f.write(f"5-Fold CV F1:        {cv_f1.mean()*100:.2f}% (+/-{cv_f1.std()*100:.2f}%)\n\n")
    f.write("--- Isolation Forest (Unsupervised Baseline) ---\n")
    f.write(f"Accuracy:            {iso_acc*100:.2f}%\n")
    f.write(f"F1-Score:            {iso_f1*100:.2f}%\n")

print(f"\n✅ Models saved to {MODELS_DIR}/")
print(f"✅ Metrics saved to {MODELS_DIR}/metrics.txt")

# ─────────────────────────────────────────────
# 9. VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IoT Predictive Maintenance — Random Forest Evaluation', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues',
            xticklabels=['Normal','Failure'], yticklabels=['Normal','Failure'])
axes[0,0].set_title('Confusion Matrix — Random Forest')
axes[0,0].set_ylabel('Actual'); axes[0,0].set_xlabel('Predicted')

# Feature Importance
feat_imp.head(8).plot(kind='barh', ax=axes[0,1], color='steelblue', edgecolor='black')
axes[0,1].set_title('Top Feature Importances')
axes[0,1].set_xlabel('Importance Score')
axes[0,1].invert_yaxis()

# Model Comparison
models = ['Random Forest', 'Isolation Forest']
f1s = [rf_f1*100, iso_f1*100]
colors = ['#2196F3', '#FF9800']
bars = axes[1,0].bar(models, f1s, color=colors, edgecolor='black', linewidth=0.8)
axes[1,0].set_title('F1-Score: Supervised vs Unsupervised')
axes[1,0].set_ylabel('F1-Score (%)')
axes[1,0].set_ylim(0, 100)
for bar, val in zip(bars, f1s):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, val+1, f'{val:.1f}%',
                   ha='center', fontweight='bold')

# Sensor Distribution — Normal vs Failure
df_plot = df.copy()
axes[1,1].boxplot([
    df_plot[df_plot['Machine failure']==0]['Tool wear [min]'].values,
    df_plot[df_plot['Machine failure']==1]['Tool wear [min]'].values
], labels=['Normal', 'Failure'])
axes[1,1].set_title('Tool Wear Distribution: Normal vs Failure')
axes[1,1].set_ylabel('Tool Wear [min]')

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'evaluation_report.png'), dpi=150, bbox_inches='tight')
print(f"✅ Charts saved to {MODELS_DIR}/evaluation_report.png")

print("\n" + "=" * 65)
print("PIPELINE COMPLETE ✅")
print("=" * 65)
