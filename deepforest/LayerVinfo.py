"""
Layer with evolutionary selective ensemble.
Uses V-information and size as two objectives for multi-objective optimization.
"""
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import geatpy as ea
from deepforest.utils import calculate_v_information


class VInfoSizeProblem(ea.Problem):
    """
    Multi-objective optimization problem for selective ensemble.
    Objective 1: Maximize V-information (minimize negative V-info)
    Objective 2: Minimize distance to target size
    """
    def __init__(self, tree_proba_list, train_labels, val_labels, num_classes, target_size=60):
        name = 'VInfoSizeProblem'
        M = 2  # 2 objectives
        maxormins = [-1, 1]  # maximize v-info, minimize size difference
        Dim = len(tree_proba_list)
        varTypes = [1] * Dim  # binary variables
        lb = [0] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.tree_proba_list = tree_proba_list
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.num_classes = num_classes
        self.target_size = target_size

    def evalVars(self, Vars):
        ObjV = []
        n_learners = len(Vars[0])
        
        for ind in range(len(Vars)):
            s = Vars[ind]
            n_selected = sum(s)
            
            # If no tree selected, give penalty
            if n_selected == 0:
                ObjV.append([-np.inf, np.inf])
                continue
            
            # Calculate ensemble prediction
            proba_sum = np.zeros_like(self.tree_proba_list[0])
            for i in range(n_learners):
                if s[i] == 1:
                    proba_sum += self.tree_proba_list[i]
            proba_avg = proba_sum / n_selected
            
            # Objective 1: V-information
            v_info, _, _ = calculate_v_information(self.train_labels, self.val_labels, proba_avg)
            
            # Objective 2: Distance to target size
            size_diff = abs(n_selected - self.target_size)
            
            ObjV.append([v_info, size_diff])
        
        return np.vstack(ObjV)


class LayerVinfo:
    """
    Layer with evolutionary selective ensemble.
    Each fold uses NSGA-II to select trees based on V-info and size.
    """
    def __init__(self, num_forests, n_estimators, num_classes,
                 n_fold, layer_index, max_depth=100, min_samples_leaf=1,
                 pop_size=100, max_gen=100, target_size=60):
        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_fold = n_fold
        self.layer_index = layer_index
        self.forest_list = []  # stores selected trees for each forest/fold
        self.fold_train_labels = []
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.target_size = target_size

    def _generate_base_learners(self, X_train, y_train, forest_index):
        """Generate base learners (trees) for a forest."""
        if forest_index % 2 == 0:
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=-1,
                max_features="sqrt"
            )
        else:
            forest = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=-1,
                max_features=1
            )
        forest.fit(X_train, y_train)
        return list(forest.estimators_)

    def _select_estimators(self, tree_list, train_labels, val_data, val_labels):
        """
        Use NSGA-II to select trees based on V-info and size.
        Returns selected trees (those with maximum V-info from Pareto front).
        """
        # Get predictions from all trees
        tree_proba_list = []
        for tree in tree_list:
            proba = tree.predict_proba(val_data)
            tree_proba_list.append(proba)
        
        # Create optimization problem
        problem = VInfoSizeProblem(
            tree_proba_list=tree_proba_list,
            train_labels=train_labels,
            val_labels=val_labels,
            num_classes=self.num_classes,
            target_size=self.target_size
        )
        
        # Run NSGA-II
        algorithm = ea.moea_NSGA2_templet(
            problem,
            ea.Population(Encoding='BG', NIND=self.pop_size),
            MAXGEN=self.max_gen,
            logTras=0
        )
        
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
        
        # Get Pareto front solutions
        pop = res['Vars']
        objv = res['ObjV']
        
        # Select solution with maximum V-info from Pareto front
        best_idx = np.argmax(objv[:, 0])  # V-info is first objective
        selected_mask = pop[best_idx]
        
        # Build selected tree list
        selected_trees = [tree_list[i] for i, flag in enumerate(selected_mask) if flag == 1]
        return selected_trees, objv[best_idx, 0]  # return trees and v-info

    def _predict_proba_from_trees(self, tree_list, X):
        """Get average prediction from a list of trees."""
        if len(tree_list) == 0:
            return np.zeros((X.shape[0], self.num_classes))
        
        proba_sum = np.zeros((X.shape[0], self.num_classes))
        for tree in tree_list:
            proba = tree.predict_proba(X)
            proba_sum += proba
        return proba_sum / len(tree_list)

    def train(self, train_data, train_label):
        """
        Train one layer with evolutionary selective ensemble.
        Returns validation predictions and v-info metrics.
        """
        num_samples = train_data.shape[0]
        val_prob = np.zeros((self.num_forests, num_samples, self.num_classes), dtype=np.float64)
        
        # V-info metrics for each forest
        forest_v_info_list = []
        forest_hv_empty_list = []
        forest_hv_cond_list = []

        for forest_index in range(self.num_forests):
            val_forest = np.zeros((num_samples, self.num_classes), dtype=np.float64)
            kfold = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
            kfold_indexes = list(kfold.split(train_data, train_label))

            fold_tree_list = []
            fold_train_labels_list = []
            
            fold_v_info_list = []
            fold_hv_empty_list = []
            fold_hv_cond_list = []

            for train_index, val_index in kfold_indexes:
                train_data_k = train_data[train_index, :]
                train_label_k = train_label[train_index]
                val_data_k = train_data[val_index, :]
                val_label_k = train_label[val_index]
                
                fold_train_labels_list.append(train_label_k)

                # Generate base learners
                base_trees = self._generate_base_learners(train_data_k, train_label_k, forest_index)
                
                # Evolutionary selection
                selected_trees, fold_v_info = self._select_estimators(
                    base_trees, train_label_k, val_data_k, val_label_k
                )
                
                fold_tree_list.append(selected_trees)
                
                # Get predictions from selected trees
                val_predict_proba = self._predict_proba_from_trees(selected_trees, val_data_k)
                val_forest[val_index, :] = val_predict_proba
                
                # Calculate v-info metrics
                v_info, hv_empty, hv_cond = calculate_v_information(train_label_k, val_label_k, val_predict_proba)
                fold_v_info_list.append(v_info)
                fold_hv_empty_list.append(hv_empty)
                fold_hv_cond_list.append(hv_cond)
                
                print(f"  Forest {forest_index}, Fold: selected {len(selected_trees)}/{len(base_trees)} trees, V-info: {v_info:.4f}")

            self.forest_list.append(fold_tree_list)
            val_prob[forest_index, :] = val_forest
            self.fold_train_labels.append(fold_train_labels_list)
            
            forest_v_info_list.append(np.mean(fold_v_info_list))
            forest_hv_empty_list.append(np.mean(fold_hv_empty_list))
            forest_hv_cond_list.append(np.mean(fold_hv_cond_list))

        val_avg = np.sum(val_prob, axis=0) / self.num_forests
        feature_new = val_prob.transpose((1, 0, 2))
        feature_new = feature_new.reshape((num_samples, -1))
        
        layer_v_info = np.mean(forest_v_info_list)
        layer_hv_empty = np.mean(forest_hv_empty_list)
        layer_hv_cond = np.mean(forest_hv_cond_list)
        
        return [val_avg, feature_new, layer_v_info, layer_hv_empty, layer_hv_cond]

    def predict(self, test_data, test_label=None):
        """
        Predict using trained layer.
        """
        num_samples = test_data.shape[0]
        test_prob = np.zeros((self.num_forests, num_samples, self.num_classes), dtype=np.float64)
        
        forest_v_info_list = []
        forest_hv_empty_list = []
        forest_hv_cond_list = []

        for forest_index in range(self.num_forests):
            test_prob_forest = np.zeros((num_samples, self.num_classes), dtype=np.float64)
            fold_tree_lists = self.forest_list[forest_index]
            
            fold_v_info_list = []
            fold_hv_empty_list = []
            fold_hv_cond_list = []
            
            for fold_index, tree_list in enumerate(fold_tree_lists):
                fold_proba = self._predict_proba_from_trees(tree_list, test_data)
                test_prob_forest += fold_proba
                
                if test_label is not None:
                    train_label_k = self.fold_train_labels[forest_index][fold_index]
                    v_info, hv_empty, hv_cond = calculate_v_information(train_label_k, test_label, fold_proba)
                    fold_v_info_list.append(v_info)
                    fold_hv_empty_list.append(hv_empty)
                    fold_hv_cond_list.append(hv_cond)

            test_prob_forest /= self.n_fold
            test_prob[forest_index, :] = test_prob_forest
            
            if test_label is not None:
                forest_v_info_list.append(np.mean(fold_v_info_list))
                forest_hv_empty_list.append(np.mean(fold_hv_empty_list))
                forest_hv_cond_list.append(np.mean(fold_hv_cond_list))

        test_avg = np.sum(test_prob, axis=0) / self.num_forests
        test_feature_new = test_prob.transpose((1, 0, 2))
        test_feature_new = test_feature_new.reshape((num_samples, -1))
        
        if test_label is not None:
            layer_v_info = np.mean(forest_v_info_list)
            layer_hv_empty = np.mean(forest_hv_empty_list)
            layer_hv_cond = np.mean(forest_hv_cond_list)
            return [test_avg, test_feature_new, layer_v_info, layer_hv_empty, layer_hv_cond]
        
        return [test_avg, test_feature_new]
