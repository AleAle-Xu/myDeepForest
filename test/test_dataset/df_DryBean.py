import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deepforest_eoh.Layer import Layer
from deepforest_eoh.gcForest import gcForest
from deepforest_eoh.utils import get_dir_in_root

path = get_dir_in_root("dataset")
path = os.path.join(path,"DryBean.csv")
dataset = pd.read_csv(path)
X = dataset.drop(columns=["target"])
y = dataset["target"]

# 将DataFrame转换为NumPy数组
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = gcForest(num_estimator=100, num_forests=4, num_classes=7, n_fold=3)
val_p, val_acc, best_layer_index = model.train(X_train, y_train)
test_p, test_acc, best_layer = model.predict(X_test, y_test)
print(test_acc[best_layer])
