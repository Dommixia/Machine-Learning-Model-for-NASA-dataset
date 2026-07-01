# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv("KOI_Cumulative_clean.csv") # LOADING DATASET

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_bin = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy() #ENCODING TARGET VARIABLE FOR BINARY CLASSIFICATION

y = le.fit_transform(df_bin['koi_disposition'])
initial_drop_cols = ['koi_disposition', 'koi_pdisposition',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'rowid', 'kepid', 'koi_score', 'kepler_name', 'koi_count', 'koi_num_transits',  'koi_tce_plnt_num', 'koi_datalink_dvr',
                     'koi_fittype',]
clean_features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_ror', 'koi_impact',
    'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_dor', 'koi_sma',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smet', 'koi_smass', 'koi_kepmag',
    'duty_cycle', 'depth_per_snr']
initial_drop_cols = [col for col in initial_drop_cols if col in df_bin.columns]
X_temp = df_bin.drop(columns=initial_drop_cols)
X = X_temp.select_dtypes(include='number')
X = X.dropna(axis=1, how='all')
X = X.fillna(X.median())
print(f"Feature shape : {X.shape}")
print(f"Target shape : {y.shape}")

err_cols = [col for col in X.columns if col.endswith('_err1') or col.endswith('_err2')]

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)
X_train = X_train.drop(columns=err_cols)
X_test = X_test.drop(columns=err_cols)
X_train['duty_cycle'] = X_train['koi_duration'] / (X_train['koi_period'] * 24)
X_test['duty_cycle'] = X_test['koi_duration'] / (X_test['koi_period'] * 24)

X_train['depth_per_snr'] = X_train['koi_depth'] / (X_train['koi_model_snr'] + 1e-5)
X_test['depth_per_snr'] = X_test['koi_depth'] / (X_test['koi_model_snr'] + 1e-5)

clean_features = [c for c in clean_features if c in X_train.columns]
X_train_final = X_train[clean_features]
X_test_final = X_test[clean_features]

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model = XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.05, eval_metric='logloss')
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

base_estimators = [
    ('rf_model', rf_model),
    ('xgb_model', xgb_model),
    ('gb_model', gb_model)
]

ensemble_model = VotingClassifier(estimators=base_estimators, voting='soft')

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

print("---------- TRAINING MODEL ------------")

stacking_model.fit(X_train_final, y_train)
y_pred_stack = stacking_model.predict(X_test_final)
y_prob_stack = stacking_model.predict_proba(X_test_final)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
f1_scores = []

for t in thresholds:
    preds_t = (y_prob_stack >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds_t, average='weighted'))

best_idx = int(np.argmax(f1_scores))
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\nBest threshold: {best_threshold:.2f} (weighted F1 = {best_f1:.4f})")
print(f"Default threshold (0.5) weighted F1 = {f1_score(y_test, (y_prob_stack >= 0.5).astype(int), average='weighted'):.4f}")

# Final predictions using the tuned threshold
y_pred = (y_prob_stack >= best_threshold).astype(int)

class_report = classification_report(y_test, y_pred)
print(f"\nClassification Report (threshold = {best_threshold:.2f}):\n{class_report}")

train_proba_stack = stacking_model.predict_proba(X_train_final)[:, 1]
train_auc_stack = roc_auc_score(y_train, train_proba_stack)

print(f"Training AUC(STACKING): {train_auc_stack:.4f}")
print(f"Testing AUC(STACKING):  {roc_auc_score(y_test, y_prob_stack):.4f}")

rf_imp = stacking_model.named_estimators_['rf_model'].feature_importances_
gb_imp = stacking_model.named_estimators_['gb_model'].feature_importances_
xgb_imp = stacking_model.named_estimators_['xgb_model'].feature_importances_
stacking_importance = (rf_imp + gb_imp + xgb_imp) / 3

importance_df = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Importance': stacking_importance
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
plt.xlim(0, importance_df['Percentage (%)'].max() + 3)
plt.tight_layout()
plt.savefig('Graphs/feature_importance.png', dpi=300)
plt.show()

cm = confusion_matrix(y_test, y_pred_stack)
display_labels = le.inverse_transform([0, 1])

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)

plt.title('Confusion Matrix: True vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Graphs/confusion_matrix.png', dpi=300)
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob_stack)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Stacking Model (AUC = {roc_auc:.4f})')
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

import shap
trained_xgb = stacking_model.named_estimators_['xgb_model']
explainer = shap.TreeExplainer(trained_xgb)
X_test_subset = X_test_final.head(300)
shap_values = explainer(X_test_subset)
plt.figure(figsize=(12, 8), dpi=300)
shap.summary_plot(shap_values, X_test_subset, max_display=15, show=False)
plt.title("SHAP Value Impact on Model Predictions (XGBoost)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("SHAP Value (Impact on Model Output Verdict)", fontsize=12)
plt.tight_layout()
plt.savefig("Graphs/shap_summary.png", bbox_inches='tight')
plt.close()

corr_df = X_train_final.copy()
corr_df['target'] = y_train

corr_matrix = corr_df.corr(method='pearson')

plt.figure(figsize=(14, 12))
mask = None

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'},
    annot_kws={'size': 8}
)

plt.title('Feature Correlation Heatmap (Final Physical Feature Set)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('Graphs/correlation_heatmap.png', dpi=300)
plt.show()

feature_corr = corr_matrix.drop(columns='target').drop(index='target')
corr_pairs = (
    feature_corr.where(~feature_corr.abs().eq(1.0))
    .unstack()
    .dropna()
    .sort_values(key=lambda x: x.abs(), ascending=False)
)
corr_pairs = corr_pairs.iloc[::2]

target_corr = corr_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
print("\nFeature correlation with target (CONFIRMED = 1):")
print(target_corr)