import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from deepforest.Layer import Layer

np.random.seed(42)

# 使用 sklearn 的 make_classification 生成多分类数据集
X, y = make_classification(
    n_samples=500, n_features=20, n_informative=15,
    n_classes=3, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def test_train():
    layer = Layer(
        num_forests=4, n_estimators=100, num_classes=3,
        n_fold=3, layer_index=0, max_depth=10, min_samples_leaf=2
    )

    val_avg, feature_new = layer.train(X_train, y_train)

    print("验证集预测结果 (val_avg):")
    print(val_avg[:5])

    print("\n新特征表示 (feature_new):")
    print(feature_new[:5])
    print(f"新特征的维度: {feature_new.shape}")


def test_train_and_predict():
    layer = Layer(
        num_forests=4, n_estimators=100, num_classes=3,
        n_fold=3, layer_index=0, max_depth=10, min_samples_leaf=2
    )

    val_avg, feature_new, test_avg, test_feature_new = layer.train_and_predict(X_train, y_train, X_test)

    print("验证集预测结果 (val_avg):")
    print(val_avg[:5])

    print("\n新特征表示 (feature_new):")
    print(feature_new[:5])
    print(f"新特征的维度: {feature_new.shape}")

    print("测试集预测结果 (test_avg):")
    print(test_avg[:5])

    print("\n新测试特征表示 (test_feature_new):")
    print(test_feature_new[:5])
    print(f"新测试特征的维度: {test_feature_new.shape}")

    avg, fea_n = layer.predict(X_test)
    print("-------------------------")
    print(avg[:5])
    print(fea_n[:5])
    # print(test_avg == avg)
    # print(test_feature_new == fea_n)

    print(np.allclose(test_avg, avg)) # 消除浮点数精读导致的细微不等问题
    print(np.allclose(test_feature_new, fea_n))


# test_train()
test_train_and_predict()
