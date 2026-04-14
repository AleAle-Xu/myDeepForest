"""
Mushroom dataset preprocessing.
- dict format: features, targets
- 8124 samples, 22 features
- 2 classes: e (edible), p (poisonous)
- Nulls: stalk-root (2480/8124 = ~30%), fill with mode (most frequent)
- All 22 columns are categorical -> one-hot encode
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
path = os.path.join(path, "Mushroom.pkl")
with open(path, "rb") as f:
    dataset = pickle.load(f)

X, y = dataset["features"], dataset["targets"]

print(f"X shape: {X.shape}")
print(f"Nulls:\n{X.isnull().sum()[X.isnull().sum()>0]}")

# stalk-root has ~30% nulls -> fill with mode
for col in X.columns:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col] = X[col].fillna(mode_val)
        print(f"  Filled {col} with mode: {mode_val}")

# veil-type has only one unique value -> drop it (no information)
for col in X.columns:
    if X[col].nunique() == 1:
        print(f"  Dropping constant column: {col}")
        X = X.drop(columns=[col])

print(f"After fill & drop: {X.shape}")
print(f"Any nulls: {X.isnull().any().any()}")

label_encoder = LabelEncoder()
y = y.to_numpy().ravel()
y = label_encoder.fit_transform(y)

# One-hot encode all categorical columns
X = pd.get_dummies(X, drop_first=False, dummy_na=False)
X = X.astype(float)

print(f"After one-hot: {X.shape}")
feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='target')
data = pd.concat([X_df, y_series], axis=1)

path = get_dir_in_root("dataset")
path = os.path.join(path, "Mushroom.csv")
data.to_csv(path, index=False)
print(f"Saved to {path}")
print(f"Classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
