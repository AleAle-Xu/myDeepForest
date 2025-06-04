from deepforest_eoh.utils import *
from deepforest_eoh.Layer import *


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
        :return:    val_p: a list of every layer's predicted probability.
                    val_acc: a list of every layer's validation accuracy.
                    best_layer_index:
        """
        train_data_raw = train_data.copy()
        layer_index = 0
        val_acc = []
        val_p = []
        best_acc = 0
        bad = 0
        best_layer_index = 0
        
        print("start training")
        while layer_index <= self.max_layer:
            layer = Layer(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, layer_index,
                          self.max_depth, 1)
            val_avg, val_prob = layer.train(train_data, train_label)
            self.layer_list.append(layer)
            accuracy = compute_accuracy(train_label, val_avg)
            print(f"Layer {layer_index} train accuracy: {accuracy}")
            val_acc.append(accuracy)
            val_p.append(val_avg)

            # train_data = np.concatenate([train_data_raw, val_prob], axis=1)
            train_data = in_model_feature_transform(train_data_raw, val_prob)
            train_data = np.float16(train_data)
            train_data = np.float64(train_data)

            if accuracy > best_acc:
                best_acc = accuracy
                best_layer_index = layer_index
                bad = 0
            else:
                bad += 1
            layer_index += 1
            if bad > self.tolerance:
                break
        self.number_of_layers = layer_index
        self.best_layer = best_layer_index
        print("training finished")
        return [val_p, val_acc, best_layer_index]

    def train_and_predict(self, train_data, train_label, test_data, test_label):
        """
        Train the gcForest.All layers are stored in the member variable layer_list.
        And then predict the probability on test data.
        :param train_data:
        :param train_label:
        :param test_data:
        :param test_label:
        :return:    val_p: a list of every layer's predicted probability on train data.
                    val_acc: a list of every layer's validation accuracy.
                    test_p: a list of every layer's predicted probability on test data.
                    test_acc: a list of every layer's test accuracy
                    best_layer_index:
        """
        train_data_raw = train_data.copy()
        test_data_raw = test_data.copy()
        layer_index = 0
        val_p = []
        val_acc = []
        best_acc = 0
        test_p = []
        test_acc = []
        bad = 0
        best_layer_index = 0

        print("start training")
        while layer_index <= self.max_layer:
            layer = Layer(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, layer_index,
                          self.max_depth, 1)
            val_avg, val_prob, test_avg, test_prob = layer.train_and_predict(train_data, train_label,
                                                                                         test_data)
            self.layer_list.append(layer)
            accuracy = compute_accuracy(train_label, val_avg)
            print(f"Layer {layer_index} train accuracy: {accuracy}")
            accuracy_test = compute_accuracy(test_label, test_avg)
            print(f"Layer {layer_index} test accuracy: {accuracy_test}")
            val_acc.append(accuracy)
            val_p.append(val_avg)
            test_acc.append(accuracy_test)
            test_p.append(test_avg)

            # train_data = np.concatenate([train_data_raw, feature_new], axis=1)
            train_data = in_model_feature_transform(train_data_raw, val_prob)
            train_data = np.float16(train_data)
            train_data = np.float64(train_data)
            test_data = in_model_feature_transform(test_data_raw, test_prob)
            test_data = np.float16(test_data)
            test_data = np.float64(test_data)

            if accuracy > best_acc:
                best_acc = accuracy
                best_layer_index = layer_index
                bad = 0
            else:
                bad += 1
            layer_index += 1
            if bad > self.tolerance:
                break
        self.number_of_layers = layer_index
        self.best_layer = best_layer_index
        print("training finished")
        return [val_p, val_acc, test_p, test_acc, best_layer_index]

    def predict(self, test_data, test_label):
        """
        use the trained gcForest to predict the probability on the test data
        :param test_data:
        :param test_label:
        :return:    test_p: a list of every layer's predicted probability on the test data.
                    test_acc: a list of every layer's test accuracy.
                    best_layer: index.
        """
        test_data_raw = test_data.copy()
        test_p = []
        test_acc = []

        print("start testing")
        for i in range(self.number_of_layers):
            model = self.layer_list[i]
            test_avg, test_prob = model.predict(test_data)
            test_p.append(test_avg)
            accuracy = compute_accuracy(test_label, test_avg)
            print(f"Layer {i} test accuracy: {accuracy}")
            test_acc.append(accuracy)
            # test_data = np.concatenate([test_data_raw, test_feature_new], axis=1)
            test_data = in_model_feature_transform(test_data_raw, test_prob)
            test_data = np.float16(test_data)
            test_data = np.float64(test_data)

        print("testing finished")
        return [test_p, test_acc, self.best_layer]
