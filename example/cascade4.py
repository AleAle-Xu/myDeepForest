"""
without update_data
"""

import numpy as np
from sklearn.model_selection import KFold
from .layer4 import *
from .utils import *
from sklearn.metrics import accuracy_score


class CascadeForest4:
    def __init__(self, num_estimator, num_forests, num_classes, max_layer=10, max_depth=4, n_fold=3, target_size=60, record_margin_history=False):
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.num_classes = num_classes
        self.layer_list = []
        self.number_of_layers = max_layer
        self.best_layer = -1
        self.min_samples_leaf = 1
        self.labeled_margin_threshold = -1
        self.unlabeled_margin_threshold = 0.7
        self.tau = 0.1
        self.screen_threshold = 1.01
        self.target_size = target_size
        self.record_margin_history = record_margin_history

    def update_data(self, X, y, unlabeled_X, X_raw, unlabeled_X_raw, val_avg, unlabeled_avg, layer):
        original_labels = np.unique(y)

        # calculate margin for labeled data
        margin = calculate_margin(val_avg, y)

        # calculate unsupervised margin for unlabeled data
        unlabeled_pred = np.argmax(unlabeled_avg, axis=1)
        unlabeled_margin = calculate_margin(unlabeled_avg, unlabeled_pred)

        # if layer <2:
        #     plot_margin_histograms(margin,unlabeled_margin,layer)

        # data to keep learning
        index_remain_l = np.logical_and(margin >= self.labeled_margin_threshold, margin < self.screen_threshold)
        remain_X_l = X[index_remain_l]
        remain_y_l = y[index_remain_l]
        remain_X_raw_l = X_raw[index_remain_l]

        # data to convert to unlabeled
        index_convert_to_u = (margin < self.labeled_margin_threshold)
        convert_X_to_u = X[index_convert_to_u]
        convert_X_raw_to_u = X_raw[index_convert_to_u]

        # update X, y and unlabeled_X
        index_convert_to_l = (unlabeled_margin > self.unlabeled_margin_threshold)
        X_new = np.concatenate((remain_X_l, unlabeled_X[index_convert_to_l]), axis=0)
        y_new = np.concatenate((remain_y_l, unlabeled_pred[index_convert_to_l]), axis=0)
        unlabeled_X_new = np.concatenate((convert_X_to_u, unlabeled_X[~index_convert_to_l]), axis=0)
        # update X_raw, unlabeled_X_raw
        X_raw_new = np.concatenate((remain_X_raw_l, unlabeled_X_raw[index_convert_to_l]), axis=0)
        unlabeled_X_raw_new = np.concatenate((convert_X_raw_to_u, unlabeled_X_raw[~index_convert_to_l]), axis=0)

        #unique_labels = np.unique(y_new)
        label_counts = {label: np.sum(y_new == label) for label in original_labels}
        #missing_labels = [label for label in original_labels if label not in unique_labels]
        labels_to_fix = [label for label, count in label_counts.items() if count< self.n_fold]

        for label in labels_to_fix:
            # find all instances that are labeled as 'label' in X and y
            # get top k(10%) of them according to margin
            # make sure that the number of samples for each category after correction is >= self.nfold
            original_indices = np.where(y == label)[0]
            label_margins = margin[original_indices]
            min_counts_to_fix = self.n_fold-label_counts[label]
            top_k = max(min_counts_to_fix, int(0.1 * len(label_margins)))
            #print(top_k)
            top_indices = original_indices[np.argsort(-label_margins)[:top_k]]

            selected_X = X[top_indices]
            selected_X_raw = X_raw[top_indices]
            selected_y = y[top_indices]

            # restore them to X_new or y_new
            X_new = np.concatenate((X_new, selected_X), axis=0)
            y_new = np.concatenate((y_new, selected_y), axis=0)
            X_raw_new = np.concatenate((X_raw_new, selected_X_raw), axis=0)


            # remove the restored instances in unlabelde_X_new
            to_remove_indices = []
            for i, item in enumerate(unlabeled_X_new):
                for selected in selected_X:
                    if np.array_equal(item, selected):
                        to_remove_indices.append(i)
                        break

            mask = np.ones(len(unlabeled_X_new), dtype=bool)
            mask[to_remove_indices] = False
            unlabeled_X_new = unlabeled_X_new[mask]
            unlabeled_X_raw_new = unlabeled_X_raw_new[mask]

        return X_new, y_new, unlabeled_X_new, X_raw_new, unlabeled_X_raw_new

    def fit(self, X, y, unlabeled_X):  # train_data, train_label
        total_n_samples = X.shape[0] + unlabeled_X.shape[0]
        X_raw = X.copy()
        unlabeled_X_raw = unlabeled_X.copy()

        # return value
        val_p = []
        val_acc = []
        margin_history = {}

        best_train_acc = 0.0
        layer_index = 0
        best_layer_index = 0
        bad = 0

        while layer_index < self.max_layer:
            print("layer " + str(layer_index), "X.shape {}, unlabeled_X.shape {}".format(X.shape, unlabeled_X.shape))
            layer = Layer4(self.num_forests, self.num_estimator, self.num_classes,
                           self.n_fold, layer_index, self.max_depth, self.min_samples_leaf, self.target_size)
            val_avg, unlabeled_avg, val_concatenate, unlabeled_concatenate, val_avg_before, unlabeled_avg_before = layer.fit(X, y, unlabeled_X)
            self.layer_list.append(layer)

            if self.record_margin_history:
                # Labeled data margins
                margin_before = calculate_margin(val_avg_before, y)
                margin_after = calculate_margin(val_avg, y)

                # Unlabeled data margins
                unlabeled_pred_before = np.argmax(unlabeled_avg_before, axis=1)
                unlabeled_margin_before = calculate_margin(unlabeled_avg_before, unlabeled_pred_before)
                unlabeled_pred_after = np.argmax(unlabeled_avg, axis=1)
                unlabeled_margin_after = calculate_margin(unlabeled_avg, unlabeled_pred_after)

                margin_history[layer_index] = {
                    'labeled': {
                        'before': margin_before.tolist(),
                        'after': margin_after.tolist()
                    },
                    'unlabeled': {
                        'before': unlabeled_margin_before.tolist(),
                        'after': unlabeled_margin_after.tolist()
                    }
                }

            X = np.concatenate([X_raw, val_concatenate], axis=1)
            X = np.float16(X)
            X = np.float64(X)

            unlabeled_X = np.concatenate([unlabeled_X_raw, unlabeled_concatenate], axis=1)
            unlabeled_X = np.float16(unlabeled_X)
            unlabeled_X = np.float64(unlabeled_X)

            temp_val_acc = accuracy_score(y, np.argmax(val_avg, axis=1))
            print('Val Accuracy:', temp_val_acc)
            val_acc.append(temp_val_acc)

            # update X and unlabeled_X. Note that X_raw and unlabeled_X_raw should be maintained, else the dim of X will keep increasing.
            X, y, unlabeled_X, X_raw, unlabeled_X_raw = self.update_data(X, y, unlabeled_X, X_raw, unlabeled_X_raw,
                                                                         val_avg, unlabeled_avg, layer_index)

            if best_train_acc > temp_val_acc:
                bad += 1
            else:
                bad = 0
                best_train_acc = temp_val_acc
                best_layer_index = layer_index
            if bad >= 2:
                self.number_of_layers = layer_index + 1
                print("bad >= 2, end")
                break

            self.best_layer = best_layer_index

            if unlabeled_X.shape[0] * 1.0 / total_n_samples < self.tau or X.shape[0] < self.n_fold * 5:
                self.number_of_layers = layer_index + 1
                if unlabeled_X.shape[0] * 1.0 / total_n_samples < self.tau:
                    print("percentage of unlabel_X < tau ")
                    print(unlabeled_X.shape[0])
                    print(total_n_samples)
                else:
                    print("number of X < n_fold*5")
                break

            layer_index = layer_index + 1

        print(best_layer_index)
        return [val_p, val_acc, best_layer_index, margin_history]

    def predict(self, test_data, test_label=None):
        test_data_raw = test_data.copy()
        layer_index = 0
        layer_accuracies = {}
        best_layer_prob = None

        # 预测所有层
        while layer_index < self.number_of_layers:
            layer = self.layer_list[layer_index]
            predict_prob = np.zeros((self.num_forests, test_data.shape[0], self.num_classes), dtype=np.float64)
            n_dim = test_data.shape[1]
            for forest_index in range(self.num_forests):
                predict_prob_forest = np.zeros([test_data.shape[0], self.num_classes])
                for kfold in range(self.n_fold):
                    tree_list = layer.forest_list[forest_index][kfold]
                    predict_p = predict_proba_from_treelist(tree_list, test_data, self.num_classes)
                    predict_prob_forest += predict_p
                predict_prob_forest /= self.n_fold
                predict_prob[forest_index, :] = predict_prob_forest
            predict_avg = np.sum(predict_prob, axis=0)
            predict_avg /= self.num_forests
            predict_concatenate = predict_prob.transpose((1, 0, 2))
            predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)

            test_prob, test_concatenate = predict_avg, predict_concatenate

            # 保存最佳层的预测结果
            if layer_index == self.best_layer:
                best_layer_prob = test_prob.copy()

            test_data = np.concatenate([test_data_raw, test_concatenate], axis=1)
            test_data = np.float16(test_data)
            test_data = np.float64(test_data)
            if test_label is not None:
                # Evaluation
                performance = accuracy_score(test_label, np.argmax(test_prob, axis=1))
                print('layer={}, test performance={}'.format(layer_index, performance))
                layer_accuracies[layer_index] = performance
            layer_index += 1

        # 返回最佳层的结果
        return best_layer_prob, layer_accuracies
