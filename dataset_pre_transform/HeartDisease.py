"""
HeartDisease dataset preprocessing.
- dict format: features, targets
- 303 samples, 13 features
- 5 classes: 0,1,2,3,4
- Nulls: ca(4), thal(2) -> few, drop rows
- No categorical columns (all numeric)
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
path = os.path.join(path, "HeartDisease.pkl")
with open(path, "rb") as f:
    dataset = pickle.load(f)

X, y = dataset["features"], dataset["targets"]

print(f"X shape: {X.shape}")
print(f"Nulls:\n{X.isnull().sum()[X.isnull().sum()>0]}")

# Few nulls (4 + 2 = 6 rows), drop rows with any null
y_series_tmp = y.squeeze()
mask = X.isnull().any(axis=1)
print(f"Dropping {mask.sum()} rows with nulls")
X = X[~mask].reset_index(drop=True)
y_series_tmp = y_series_tmp[~mask.values].reset_index(drop=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_series_tmp.to_numpy().ravel())

X = X.astype(float)
feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='target')
data = pd.concat([X_df, y_series], axis=1)

path = get_dir_in_root("dataset")
path = os.path.join(path, "HeartDisease.csv")
data.to_csv(path, index=False)
print(f"Saved to {path}")
print(f"Final shape: {data.shape}")
print(f"Classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
