import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("KOI_Cumulative_clean.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_bin = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
y = le.fit_transform(df_bin['koi_disposition'])
drop_cols = [
    'koi_disposition', 'kepid', 'kepoi_name', 'kepler_name',
    'koi_vet_stat', 'koi_vet_date', 'koi_pdisposition',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]
drop_cols = [col for col in drop_cols if col in df_bin.columns]
X = df_bin.drop(columns=drop_cols)
print(f"Feature shape : {X.shape}")
print(f"Target shape : {y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
