"""
Car dataset preprocessing.
- dict format: features, targets
- 1728 samples, 6 features
- 4 classes: acc, good, unacc, vgood
- No nulls
- All 6 columns are categorical -> one-hot encode
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
path = os.path.join(path, "Car.pkl")
with open(path, "rb") as f:
    dataset = pickle.load(f)

X, y = dataset["features"], dataset["targets"]

print(X.head())
print(X.dtypes)
print(f"Any nulls: {X.isnull().any().any()}")

label_encoder = LabelEncoder()
y = y.to_numpy().ravel()
y = label_encoder.fit_transform(y)

# All columns are categorical: one-hot encode
X = pd.get_dummies(X, drop_first=False, dummy_na=False)
X = X.astype(float)

print(f"After one-hot X shape: {X.shape}")

feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='target')
data = pd.concat([X_df, y_series], axis=1)

path = get_dir_in_root("dataset")
path = os.path.join(path, "Car.csv")
data.to_csv(path, index=False)
print(f"Saved to {path}")
print(f"Classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
