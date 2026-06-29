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

print("Training model")
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

print("TOP PHYSICAL FEATURES AND THEIR WEIGHTS")
print(importance_df.head(15).to_string(index=True, formatters={
    'Importance': '{:,.4f}'.format,
    'Percentage (%)': '{:,.2f}%'.format,
    'Cumulative Percentage (%)': '{:,.2f}%'.format
}))