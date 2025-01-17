from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np


class Layer:
    def __init__(self, num_forests, n_estimators, num_classes,
                 n_fold, layer_index, max_depth=100, min_samples_leaf=1):
        self.num_forests = num_forests  # number of forests
        self.n_estimators = n_estimators  # number of trees in each forest
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_fold = n_fold
        self.layer_index = layer_index
        self.forest_list = []

    def train(self, train_data, train_label):
        """
        Train one layer of the gcForest.All trained models(forests) are stored in the member variable forest_list
        :param train_data:
        :param train_label:
        :return: val_avg: The final prediction results for train_data.It's the average probability predicted by all forests.
                 feature_new: New features to pass to next layer
        """
        num_samples = train_data.shape[0]
        val_prob = np.zeros((self.num_forests, num_samples, self.num_classes),dtype=np.float64)

        for forest_index in range(self.num_forests):
            val_forest = np.zeros((num_samples, self.num_classes),dtype=np.float64)
            kfold = KFold(n_splits=self.n_fold, shuffle=True)
            kfold_indexes = []  # stores k lists representing k pairs of indexes like (train_index,val_index)

            for i, j in kfold.split(range(num_samples)):
                kfold_indexes.append([i, j])

            model_list = []  # one forest corresponding to k models

            if forest_index % 2 == 0:  # train a RandomForestClassifier
                for train_index, val_index in kfold_indexes:
                    train_data_k = train_data[train_index, :]
                    train_label_k = train_label[train_index]
                    val_data_k = train_data[val_index, :]
                    val_label_k = train_label[val_index]

                    model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                   min_samples_leaf=self.min_samples_leaf, n_jobs=-1,
                                                   max_features="sqrt")
                    model.fit(train_data_k, train_label_k)
                    model_list.append(model)
                    val_predict_proba = model.predict_proba(val_data_k)
                    val_forest[val_index, :] = val_predict_proba  # every loop gets one fold's data's prediction probability

            else:
                for train_index, val_index in kfold_indexes:
                    train_data_k = train_data[train_index, :]
                    train_label_k = train_label[train_index]
                    val_data_k = train_data[val_index, :]
                    val_label_k = train_label[val_index]

                    model = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                 min_samples_leaf=self.min_samples_leaf, n_jobs=-1,
                                                 max_features=1)
                    model.fit(train_data_k, train_label_k)
                    model_list.append(model)
                    val_predict_proba = model.predict_proba(val_data_k)
                    val_forest[val_index, :] = val_predict_proba

            self.forest_list.append(model_list)
            val_prob[forest_index, :] = val_forest

        val_avg = np.sum(val_prob, axis=0) / self.num_forests
        feature_new = val_prob.transpose((1, 0, 2))
        feature_new = feature_new.reshape((num_samples, -1))
        return [val_avg, feature_new]
