import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from deepforest.Layer import Layer
from deepforest.gcForest import gcForest

#np.random.seed(42)

# 使用 sklearn 的 make_classification 生成多分类数据集
X, y = make_classification(
    n_samples=500, n_features=20, n_informative=15,
    n_classes=3
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def test_train():
    model = gcForest(num_estimator=100, num_forests=4, num_classes=3)
    val_p, val_acc, best_layer_index = model.train(X_train, y_train)
    print(val_acc)
    print(best_layer_index)

def test_train_and_predict():
    model = gcForest(num_estimator=100, num_forests=4, num_classes=3)
    val_p, val_acc, test_p, test_acc, best_layer_index = model.train_and_predict(X_train, y_train, X_test, y_test)
    print(val_p[0].shape)
    print(len(val_p))
    print(model.number_of_layers)
    print(val_acc)
    print(test_p[0].shape)
    print(test_acc)
    print(best_layer_index)


#test_train()
test_train_and_predict()
