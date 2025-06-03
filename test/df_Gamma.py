import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from deepforest_eoh.Layer import Layer
from deepforest_eoh.gcForest import gcForest
from deepforest_eoh.utils import get_dir_in_root

path = get_dir_in_root("dataset")
path = os.path.join(path,"Gamma.pkl")
with open(path,"rb") as f:
    dataset = pickle.load(f)

X, y = dataset["features"], dataset["targets"]  # DataFrame DataFrame
X = X.to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 处理目标变量 y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.to_numpy().ravel())  # ndarray


X_train, X_test, y_train, y_test = train_test_split(X,y)

model = gcForest(num_estimator=100, num_forests=4, num_classes=2, n_fold=3)
val_p, val_acc, best_layer_index = model.train(X_train, y_train)
test_p, test_acc, best_layer = model.predict(X_test, y_test)
print(test_acc[best_layer])
