import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv("KOI_Cumulative_clean.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_bin = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
y = le.fit_transform(df_bin['koi_disposition'])
initial_drop_cols = [
'koi_disposition', 'koi_pdisposition',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]
initial_drop_cols = [col for col in initial_drop_cols if col in df_bin.columns]
X_temp = df_bin.drop(columns=initial_drop_cols)
X = X_temp.select_dtypes(include='number')
X = X.dropna(axis=1, how='all')
X = X.fillna(X.median())
print(f"Feature shape : {X.shape}")
print(f"Target shape : {y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model = XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.05, eval_metric='logloss')
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

ensemble_model = VotingClassifier([
    ('rf_model', rf_model),
    ('xgb_model', xgb_model),
    ('gb_model', gb_model)
], voting='soft')

print("----------------------------------------")

print("---------- TRAINING MODEL ------------")

train_start = time.perf_counter()
ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)
y_prob = ensemble_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
class_report = classification_report(y_test, y_pred)
train_end = time.perf_counter()
time = train_end - train_start
train_proba = ensemble_model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_proba)

print(f"Training AUC: {train_auc:.4f}")
print(f"Testing AUC:  {roc_auc_score(y_test, y_prob):.4f}")

print(f"Classification Report: {class_report}")
print(f"AUC: {roc_auc}")

print(f"Time Taken to Train Model: {time}")

rf_imp = ensemble_model.named_estimators_['rf_model'].feature_importances_
gb_imp = ensemble_model.named_estimators_['gb_model'].feature_importances_
xgb_imp = ensemble_model.named_estimators_['xgb_model'].feature_importances_
ensemble_importance = (rf_imp + gb_imp + xgb_imp) / 3
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': ensemble_importance
})

importance_df['Percentage (%)'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
importance_df['Cumulative Percentage (%)'] = importance_df['Percentage (%)'].cumsum()

import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, confusion_matrix

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Percentage (%)',
    y='Feature',
    data=importance_df.head(15),
    palette='viridis',
    hue='Feature',
    legend=False
)

for index, row in importance_df.head(15).iterrows():
    plt.text(row['Percentage (%)'] + 0.2, index, f"{row['Percentage (%)']:.1f}%",
             va='center', fontsize=10, fontweight='bold')

plt.title('Top 15 Most Important Physical Features (% of Model Weight)', fontsize=14, fontweight='bold')
plt.xlabel('Predictive Weight Percentage (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, importance_df['Percentage (%)'].max() + 3) # Give extra room for the text labels
plt.tight_layout()
plt.savefig('Graphs/feature_importance.png', dpi=300)
plt.show()

cm = confusion_matrix(y_test, y_pred)
display_labels = le.inverse_transform([0, 1]) # Maps 0 and 1 back to 'CONFIRMED' / 'FALSE POSITIVE'

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)

plt.title('Confusion Matrix: True vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Graphs/confusion_matrix.png', dpi=300)
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Ensemble Voting Model (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guessing (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Graphs/roc_curve.png', dpi=300)
plt.show()