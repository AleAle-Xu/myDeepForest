import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from deepforest_eoh.utils import get_dir_in_root

pd.set_option('display.max_columns', None)

path = get_dir_in_root("dataset_raw")
path = os.path.join(path,"Adult.pkl")
with open(path,"rb") as f:
    dataset = pickle.load(f)

X, y = dataset.data, dataset.target
X = pd.get_dummies(X,dummy_na=True)
X = X.astype(float)
# print(X.isna().any().any())
# print(X)
#print(y.isna().any())

# 保存原始列名
feature_names = X.columns.tolist()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # ndarray

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
path = os.path.join(path,"Adult.csv")
data.to_csv(path, index=False)

