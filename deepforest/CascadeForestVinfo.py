"""
CascadeForest with evolutionary selective ensemble.
Early stopping based on V-information change.
"""
import numpy as np
from sklearn.metrics import accuracy_score
from deepforest.LayerVinfo import LayerVinfo


class CascadeForestVinfo:
    """
    Cascade Forest with evolutionary selective ensemble.
    Uses V-information for early stopping.
    """
    def __init__(self, num_estimator=100, num_forests=4, num_classes=2,
                 max_layer=100, max_depth=31, n_fold=3, tolerance=3,
                 pop_size=100, max_gen=100, target_size=60):
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.num_classes = num_classes
        self.layer_list = []
        self.best_layer = 0
        self.min_samples_leaf = 1
        self.tolerance = tolerance
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.target_size = target_size
        
        # Metrics storage
        self.val_acc_list = []
        self.v_info_dict = {'v_info': [], 'hv_empty': [], 'hv_cond': []}

    def train(self, train_data, train_label):
        """
        Train the cascade forest with evolutionary selective ensemble.
        Early stopping based on V-information.
        """
        X_train = train_data.copy()
        X_train_raw = train_data.copy()
        
        best_v_info = -np.inf
        bad_count = 0
        layer_index = 0

        while layer_index < self.max_layer:
            print(f"\n=== Layer {layer_index} ===")
            
            layer = LayerVinfo(
                num_forests=self.num_forests,
                n_estimators=self.num_estimator,
                num_classes=self.num_classes,
                n_fold=self.n_fold,
                layer_index=layer_index,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                pop_size=self.pop_size,
                max_gen=self.max_gen,
                target_size=self.target_size
            )
            
            val_avg, feature_new, layer_v_info, layer_hv_empty, layer_hv_cond = layer.train(X_train, train_label)
            self.layer_list.append(layer)
            
            # Store metrics
            self.v_info_dict['v_info'].append(layer_v_info)
            self.v_info_dict['hv_empty'].append(layer_hv_empty)
            self.v_info_dict['hv_cond'].append(layer_hv_cond)
            
            # Calculate accuracy
            val_pred = np.argmax(val_avg, axis=1)
            val_acc = accuracy_score(train_label, val_pred) * 100
            self.val_acc_list.append(val_acc)
            
            print(f"Layer {layer_index}: Val Acc = {val_acc:.2f}%, V-info = {layer_v_info:.4f}")
            
            # Early stopping based on V-information
            if layer_v_info > best_v_info:
                best_v_info = layer_v_info
                self.best_layer = layer_index
                bad_count = 0
            else:
                bad_count += 1
            
            if bad_count >= self.tolerance:
                print(f"Early stopping at layer {layer_index}: V-info not improving for {self.tolerance} layers")
                break
            
            # Prepare features for next layer
            X_train = np.concatenate([X_train_raw, feature_new], axis=1)
            layer_index += 1

        print(f"\nTraining completed. Best layer: {self.best_layer}")
        return self.best_layer

    def test(self, test_data, test_label):
        """
        Test the cascade forest.
        Returns accuracy and per-layer metrics.
        """
        X_test = test_data.copy()
        X_test_raw = test_data.copy()
        
        test_acc_list = []
        test_v_info_dict = {'v_info': [], 'hv_empty': [], 'hv_cond': []}
        best_layer_pred = None

        for layer_index, layer in enumerate(self.layer_list):
            result = layer.predict(X_test, test_label)
            
            if test_label is not None:
                test_avg, test_feature_new, layer_v_info, layer_hv_empty, layer_hv_cond = result
                test_v_info_dict['v_info'].append(layer_v_info)
                test_v_info_dict['hv_empty'].append(layer_hv_empty)
                test_v_info_dict['hv_cond'].append(layer_hv_cond)
            else:
                test_avg, test_feature_new = result
            
            # Calculate accuracy
            test_pred = np.argmax(test_avg, axis=1)
            if test_label is not None:
                test_acc = accuracy_score(test_label, test_pred) * 100
                test_acc_list.append(test_acc)
                print(f"Layer {layer_index}: Test Acc = {test_acc:.2f}%")
            
            # Save best layer prediction
            if layer_index == self.best_layer:
                best_layer_pred = test_pred.copy()
            
            # Prepare features for next layer
            X_test = np.concatenate([X_test_raw, test_feature_new], axis=1)

        # Final accuracy at best layer
        if test_label is not None:
            best_acc = accuracy_score(test_label, best_layer_pred) * 100
            print(f"\nBest layer ({self.best_layer}) Test Acc: {best_acc:.2f}%")
            return best_acc, test_acc_list, test_v_info_dict
        
        return best_layer_pred
