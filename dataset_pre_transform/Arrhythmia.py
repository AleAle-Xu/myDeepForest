"""
Arrhythmia dataset preprocessing.
- sklearn Bunch format: d.data, d.target
- 452 samples, 279 features
- 13 classes
- Has nulls: T(8), P(22), QRST(1), J(376), heartrate(1)
  - J has 376/452 nulls (~83%), fill with median
  - others are few, fill with median too
- Categorical columns: sex and many binary waveform-exists columns -> already binary 0/1
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepforest.utils import get_dir_in_root

pd.set_option('display.max_columns', None)

path = get_dir_in_root("dataset_raw")
path = os.path.join(path, "Arrhythmia.pkl")
with open(path, "rb") as f:
    dataset = pickle.load(f)

X, y = dataset.data, dataset.target

# X is a DataFrame with mixed types; categorical columns are already binary-like (0/1/'?')
# Fill nulls: J has many nulls (376/452), fill all with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if X[col].dtype == object:
            # convert to numeric first
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())

# Categorical columns (sex, waveExists) are binary (0/1 or similar), convert to float
# Convert all to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# After numeric conversion there may still be NaN from non-numeric, fill with 0
X = X.fillna(0)
X = X.astype(float)

print(f"X shape: {X.shape}")
print(f"Any nulls: {X.isnull().any().any()}")

feature_names = X.columns.tolist()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='target')
data = pd.concat([X_df, y_series], axis=1)

path = get_dir_in_root("dataset")
path = os.path.join(path, "Arrhythmia.csv")
data.to_csv(path, index=False)
print(f"Saved to {path}")
print(f"Classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
