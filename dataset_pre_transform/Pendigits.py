"""
Pendigits dataset preprocessing (from libsvm).
- libsvm format: already numeric, no missing values
- train: 7494 samples, test: 3498 samples, 16 features, 10 classes (digits 0-9)
- Merge train and test into one full dataset (no split saved).
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

X_tr, y_tr = load_svmlight_file(os.path.join(raw_dir, "pendigits", "pendigits"))
X_t, y_t = load_svmlight_file(os.path.join(raw_dir, "pendigits", "pendigits.t"), n_features=X_tr.shape[1])

X = np.vstack([X_tr.toarray(), X_t.toarray()])
y = np.concatenate([y_tr, y_t])

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

out_path = os.path.join(get_dir_in_root("dataset"), "Pendigits.csv")
data.to_csv(out_path, index=False)
print(f"Saved to {out_path}, shape: {data.shape}, classes: {len(np.unique(y))}")
