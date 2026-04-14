"""
Maternal dataset preprocessing.
- dict format: features, targets
- 1014 samples, 6 features
- 3 classes: high risk, low risk, mid risk
- No nulls, no categorical columns (all numeric)
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
path = os.path.join(path, "Maternal.pkl")
with open(path, "rb") as f:
    dataset = pickle.load(f)

X, y = dataset["features"], dataset["targets"]

print(f"X shape: {X.shape}")
print(f"Any nulls: {X.isnull().any().any()}")

label_encoder = LabelEncoder()
y = y.to_numpy().ravel()
y = label_encoder.fit_transform(y)

X = X.astype(float)
feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='target')
data = pd.concat([X_df, y_series], axis=1)

path = get_dir_in_root("dataset")
path = os.path.join(path, "Maternal.csv")
data.to_csv(path, index=False)
print(f"Saved to {path}")
print(f"Classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
