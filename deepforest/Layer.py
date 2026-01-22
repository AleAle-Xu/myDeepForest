from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from deepforest.utils import calculate_v_information


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
        self.fold_train_labels = []  # stores training labels for each fold of each forest

    def train(self, train_data, train_label):
        """
        Train one layer of the gcForest.All trained models(forests) are stored in the member variable forest_list
        :param train_data:
        :param train_label:
        :return: val_avg: The final prediction results for train_data.It's the average probability predicted by all forests.
                 feature_new: New features to pass to next layer
        """
        num_samples = train_data.shape[0]
        val_prob = np.zeros((self.num_forests, num_samples, self.num_classes), dtype=np.float64)
        
        # v-information metrics for each forest
        forest_v_info_list = []
        forest_hv_empty_list = []
        forest_hv_cond_list = []

        for forest_index in range(self.num_forests):
            val_forest = np.zeros((num_samples, self.num_classes), dtype=np.float64)
            kfold = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
            kfold_indexes = []  # stores k lists representing k pairs of indexes like (train_index,val_index)

            for train_index, val_index in kfold.split(train_data, train_label):
                kfold_indexes.append([train_index, val_index])

            model_list = []  # one forest corresponding to k models
            fold_train_labels_list = []  # stores training labels for each fold
            
            # v-information metrics for each fold
            fold_v_info_list = []
            fold_hv_empty_list = []
            fold_hv_cond_list = []

            for train_index, val_index in kfold_indexes:
                train_data_k = train_data[train_index, :]
                train_label_k = train_label[train_index]
                val_data_k = train_data[val_index, :]
                val_label_k = train_label[val_index]
                
                # save training labels for this fold
                fold_train_labels_list.append(train_label_k)

                if forest_index % 2 == 0:
                    model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                   min_samples_leaf=self.min_samples_leaf, n_jobs=-1,
                                                   max_features="sqrt")
                else:
                    model = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                 min_samples_leaf=self.min_samples_leaf, n_jobs=-1,
                                                 max_features=1)
                model.fit(train_data_k, train_label_k)
                model_list.append(model)
                val_predict_proba = model.predict_proba(val_data_k)
                val_forest[val_index, :] = val_predict_proba
                
                # calculate v-information for this fold
                v_info, hv_empty, hv_cond = calculate_v_information(train_label_k, val_label_k, val_predict_proba)
                fold_v_info_list.append(v_info)
                fold_hv_empty_list.append(hv_empty)
                fold_hv_cond_list.append(hv_cond)

            self.forest_list.append(model_list)
            val_prob[forest_index, :] = val_forest
            
            # save training labels for each fold of this forest
            self.fold_train_labels.append(fold_train_labels_list)
            
            # average v-information metrics across all folds for this forest
            forest_v_info_list.append(np.mean(fold_v_info_list))
            forest_hv_empty_list.append(np.mean(fold_hv_empty_list))
            forest_hv_cond_list.append(np.mean(fold_hv_cond_list))

        val_avg = np.sum(val_prob, axis=0) / self.num_forests
        feature_new = val_prob.transpose((1, 0, 2))
        feature_new = feature_new.reshape((num_samples, -1))
        
        # average v-information metrics across all forests for this layer
        layer_v_info = np.mean(forest_v_info_list)
        layer_hv_empty = np.mean(forest_hv_empty_list)
        layer_hv_cond = np.mean(forest_hv_cond_list)
        
        return [val_avg, feature_new, layer_v_info, layer_hv_empty, layer_hv_cond]

    def train_and_predict(self, train_data, train_label, test_data):
        """
        Train one layer of the gcForest.All trained models(forests) are stored in the member variable forest_list.
        And then predict the probability on the test data.
        :param train_data:
        :param train_label:
        :param test_data:
        :return: val_avg: The final prediction results for train_data.It's the average probability predicted by all forests.
                 feature_new: New features of the training data to pass to next layer.
                 test_avg: The prediction results for test_data.
                 test_feature_new: New features of the test data to pass to next layer.
        """
        num_samples_train = train_data.shape[0]
        num_samples_test = test_data.shape[0]
        val_prob = np.zeros((self.num_forests, num_samples_train, self.num_classes), dtype=np.float64)
        test_prob = np.zeros((self.num_forests, num_samples_test, self.num_classes), dtype=np.float64)

        for forest_index in range(self.num_forests):
            val_forest = np.zeros((num_samples_train, self.num_classes), dtype=np.float64)
            test_forest = np.zeros((num_samples_test, self.num_classes), dtype=np.float64)
            kfold = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
            kfold_indexes = []  # stores k lists representing k pairs of indexes like (train_index,val_index)

            for train_index, val_index in kfold.split(train_data, train_label):
                kfold_indexes.append([train_index, val_index])

            model_list = []  # one forest corresponding to k models

            for train_index, val_index in kfold_indexes:
                train_data_k = train_data[train_index, :]
                train_label_k = train_label[train_index]
                val_data_k = train_data[val_index, :]

                if forest_index % 2 == 0:
                    model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                   min_samples_leaf=self.min_samples_leaf, n_jobs=-1,
                                                   max_features="sqrt")
                else:
                    model = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                 min_samples_leaf=self.min_samples_leaf, n_jobs=-1,
                                                 max_features=1)
                model.fit(train_data_k, train_label_k)
                model_list.append(model)
                val_predict_proba = model.predict_proba(val_data_k)
                val_forest[val_index, :] = val_predict_proba
                test_forest += model.predict_proba(test_data)

            self.forest_list.append(model_list)
            val_prob[forest_index, :] = val_forest
            test_forest /= self.n_fold
            test_prob[forest_index, :] = test_forest

        val_avg = np.sum(val_prob, axis=0) / self.num_forests
        feature_new = val_prob.transpose((1, 0, 2))
        feature_new = feature_new.reshape((num_samples_train, -1))
        test_avg = np.sum(test_prob, axis=0) / self.num_forests
        test_feature_new = test_prob.transpose((1, 0, 2))
        test_feature_new = test_feature_new.reshape((test_data.shape[0], -1))
        return [val_avg, feature_new, test_avg, test_feature_new]

    def predict(self, test_data, test_label=None):
        """
        use the trained layer to predict the probability on the test data
        :param test_data:
        :param test_label: optional, if provided, v-information metrics will be calculated
        :return: test_avg: The final prediction results for test_data.It's the average probability predicted by all forests.
                 test_feature_new: New features of the test data to pass to next layer.
                 layer_v_info, layer_hv_empty, layer_hv_cond: (only if test_label is provided)
        """
        num_samples = test_data.shape[0]
        test_prob = np.zeros((self.num_forests, num_samples, self.num_classes), dtype=np.float64)
        
        # v-information metrics for each forest (only if test_label is provided)
        forest_v_info_list = []
        forest_hv_empty_list = []
        forest_hv_cond_list = []

        for forest_index in range(self.num_forests):
            test_prob_forest = np.zeros((num_samples, self.num_classes), dtype=np.float64)
            forest_model = self.forest_list[forest_index]
            
            # v-information metrics for each fold
            fold_v_info_list = []
            fold_hv_empty_list = []
            fold_hv_cond_list = []
            
            for fold_index, model in enumerate(forest_model):
                model_prob = model.predict_proba(test_data)
                test_prob_forest += model_prob
                
                # calculate v-information for this fold if test_label is provided
                if test_label is not None:
                    train_label_k = self.fold_train_labels[forest_index][fold_index]
                    v_info, hv_empty, hv_cond = calculate_v_information(train_label_k, test_label, model_prob)
                    fold_v_info_list.append(v_info)
                    fold_hv_empty_list.append(hv_empty)
                    fold_hv_cond_list.append(hv_cond)

            test_prob_forest /= self.n_fold
            test_prob[forest_index, :] = test_prob_forest
            
            # average v-information metrics across all folds for this forest
            if test_label is not None:
                forest_v_info_list.append(np.mean(fold_v_info_list))
                forest_hv_empty_list.append(np.mean(fold_hv_empty_list))
                forest_hv_cond_list.append(np.mean(fold_hv_cond_list))

        test_avg = np.sum(test_prob, axis=0) / self.num_forests
        test_feature_new = test_prob.transpose((1, 0, 2))
        test_feature_new = test_feature_new.reshape((num_samples, -1))
        
        if test_label is not None:
            # average v-information metrics across all forests for this layer
            layer_v_info = np.mean(forest_v_info_list)
            layer_hv_empty = np.mean(forest_hv_empty_list)
            layer_hv_cond = np.mean(forest_hv_cond_list)
            return [test_avg, test_feature_new, layer_v_info, layer_hv_empty, layer_hv_cond]
        
        return [test_avg, test_feature_new]
