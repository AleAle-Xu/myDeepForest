"""
Segment dataset preprocessing (from libsvm).
- libsvm format: already numeric, no missing values, already scaled
- 2310 samples, 19 features, 7 classes (image segments)
- Single file (no train/test split).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepforest.utils import get_dir_in_root

raw_dir = get_dir_in_root("dataset_raw")

X, y = load_svmlight_file(os.path.join(raw_dir, "segment.scale"))
X = X.toarray()

print(f"X shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

feature_names = [f"f{i}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='target')
data = pd.concat([X_df, y_series], axis=1)

out_path = os.path.join(get_dir_in_root("dataset"), "Segment.csv")
data.to_csv(out_path, index=False)
print(f"Saved to {out_path}, shape: {data.shape}, classes: {len(np.unique(y))}")
