import os
import pickle
from calendar import month

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from deepforest_eoh.utils import get_dir_in_root

pd.set_option('display.max_columns', None)

path = get_dir_in_root("dataset_raw")
path = os.path.join(path, "Gamma.pkl")
with open(path, "rb") as f:
    dataset = pickle.load(f)

X, y = dataset["features"], dataset["targets"]

label_encoder = LabelEncoder()
y = y.to_numpy().ravel()
y = label_encoder.fit_transform(y)  # ndarray

# print(X)
# print(X.isna().any().any())

# 保存原始列名
feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将X转换为DataFrame，使用原始列名
X_df = pd.DataFrame(X, columns=feature_names)

# 将y转换为Series并设置列名
y_series = pd.Series(y, name='target')

# 合并X和y
data = pd.concat([X_df, y_series], axis=1)

# 保存为CSV文件
path = get_dir_in_root("dataset")
path = os.path.join(path,"Gamma.csv")
data.to_csv(path, index=False)
