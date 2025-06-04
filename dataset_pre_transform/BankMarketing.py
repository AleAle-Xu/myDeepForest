import os
import pickle
from calendar import month

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from deepforest_eoh.utils import get_dir_in_root

pd.set_option('display.max_columns', None)

path = get_dir_in_root("dataset_raw")
path = os.path.join(path, "BankMarketing.pkl")
with open(path, "rb") as f:
    dataset = pickle.load(f)

X, y = dataset["features"], dataset["targets"]
month_map_abbr = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
X['month'] = X['month'].map(month_map_abbr)

binary_columns = ['default', 'housing', 'loan']  # 转换为 0/1
categorical_columns = ['job', 'marital', 'education', 'contact', 'poutcome']  # 独热
numerical_columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day_of_week']  # 保留
# date_columns = ['month']

label_encoder = LabelEncoder()
y = y.to_numpy().ravel()
y = label_encoder.fit_transform(y)  # ndarray

dummy_binary = pd.get_dummies(X[binary_columns], drop_first=True, dummy_na=True)
dummy_category = pd.get_dummies(X[categorical_columns], drop_first=False, dummy_na=True)

X = pd.concat([
    X[numerical_columns],
    X["month"],
    dummy_binary,
    dummy_category,
], axis=1)

X = X.astype(float)
print(X)

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
path = os.path.join(path,"BankMarketing.csv")
data.to_csv(path, index=False)
