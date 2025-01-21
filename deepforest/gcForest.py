from deepforest.utils import *
from deepforest.Layer import *


class gcForest:
    def __init__(self, num_estimator, num_forests, num_classes, max_layer=100, max_depth=31, n_fold=5, tolerance=3):
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.num_classes = num_classes
        self.layer_list = []
        self.number_of_layers = max_layer
        self.best_layer = -1
        self.tolerance = tolerance  # times allowed that current layer's accuracy is lower than the previous best layer

    def train(self, train_data, train_label):
        """
        Train the gcForest.All layers are stored in the member variable layer_list.
        :param train_data:
        :param train_label:
        :return:    val_acc: a list of every layer's validation accuracy
                    best_layer_index:
        """
        train_data_raw = train_data.copy()
        layer_index = 0
        val_acc = []
        best_acc = 0
        bad = 0
        best_layer_index = 0

        while layer_index <= self.max_layer:
            layer = Layer(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, layer_index,
                          self.max_depth, 1)
            val_prob, feature_new = layer.train(train_data,train_label)
            self.layer_list.append(layer)
            accuracy = compute_accuracy(train_label, val_prob)
            val_acc.append(accuracy)

            train_data = np.concatenate([train_data_raw, feature_new], axis=1)
            train_data = np.float16(train_data)
            train_data = np.float64(train_data)

            if accuracy > best_acc:
                best_acc = accuracy
                best_layer_index = layer_index
                bad = 0
            else:
                bad+=1
                if bad >self.tolerance:
                    break
            layer_index+=1
            self.best_layer = best_layer_index
        return [val_acc,best_layer_index]