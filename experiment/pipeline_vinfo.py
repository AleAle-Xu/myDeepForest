"""
Pipeline for running experiments with CascadeForestVinfo (evolutionary selective ensemble).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deepforest.CascadeForestVinfo import CascadeForestVinfo

# Dataset list - all datasets that can be run
DATASETS = [
    'Gamma', 'DryBean', 'CredictCard', 'BankMarketing', 'Adult', 'Diabetes',
    'HTRU2', 'Rice', 'Mushroom', 'Websites', 'Letter', 'Car',
    'Maternal', 'Student', 'HeartDisease', 'Covertype'
]

# CascadeForestVinfo configuration
VINFO_CONFIG = {
    'num_estimator': 100,   # 100 trees per forest
    'num_forests': 4,       # 4 forests per layer
    'max_layer': 10,
    'max_depth': 10,
    'n_fold': 3,
    'tolerance': 3,
    'pop_size': 100,        # EA population size
    'max_gen': 100,         # EA generations
    'target_size': 50       # select 50 trees from 100
}

# Experiment configuration
NUM_RUNS = 5
TEST_SIZE = 0.3

# Model class name
MODEL_NAME = 'CascadeForestVinfo'


def generate_result_filename(model_name, dataset_name, config):
    """Generate result filename: ModelName_DatasetName_param1_param2_..."""
    params = [f"{k}{v}" for k, v in config.items()]
    param_str = '_'.join(params)
    return f"{model_name}_{dataset_name}_{param_str}.csv"


def load_dataset(dataset_name):
    """Load dataset from csv file."""
    dataset_dir = os.path.join(project_root, 'dataset')
    file_path = os.path.join(dataset_dir, f'{dataset_name}.csv')

    df = pd.read_csv(file_path)
    X = df.drop(columns=['target']).to_numpy()
    y = df['target'].to_numpy()
    num_classes = len(np.unique(y))

    return X, y, num_classes


def get_layer_avg_ensemble_size(layer):
    """
    Calculate average number of selected trees per forest-fold combination in a layer.
    layer.forest_list: list of num_forests elements,
      each element: list of n_fold elements,
        each element: list of selected trees
    Returns: mean count of trees across all forest-fold combinations.
    """
    counts = []
    for fold_tree_lists in layer.forest_list:  # per forest
        for tree_list in fold_tree_lists:        # per fold
            counts.append(len(tree_list))
    if counts:
        return float(np.mean(counts))
    return 0.0


def run_experiment(dataset_name, run_id, random_state):
    """Run a single experiment on a dataset."""
    X, y, num_classes = load_dataset(dataset_name)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    # Create model
    model = CascadeForestVinfo(
        num_estimator=VINFO_CONFIG['num_estimator'],
        num_forests=VINFO_CONFIG['num_forests'],
        num_classes=num_classes,
        max_layer=VINFO_CONFIG['max_layer'],
        max_depth=VINFO_CONFIG['max_depth'],
        n_fold=VINFO_CONFIG['n_fold'],
        tolerance=VINFO_CONFIG['tolerance'],
        pop_size=VINFO_CONFIG['pop_size'],
        max_gen=VINFO_CONFIG['max_gen'],
        target_size=VINFO_CONFIG['target_size']
    )

    # Train
    train_start = time.time()
    best_layer = model.train(X_train, y_train)
    train_end = time.time()
    training_time = train_end - train_start
    early_stop_training_time = model.early_stop_training_time

    # Compute per-layer avg ensemble size
    layer_avg_sizes = [get_layer_avg_ensemble_size(layer) for layer in model.layer_list]

    # Test
    test_start = time.time()
    test_acc, test_acc_list, test_v_info_dict = model.test(X_test, y_test)
    testing_time = time.time() - test_start
    best_layer_testing_time = model.best_layer_testing_time

    # Compute per-layer macro-F1 by replaying layer predictions on test set
    X_test_tmp = X_test.copy()
    X_test_raw = X_test.copy()
    test_layer_macro_f1 = []
    macro_f1 = None
    for i, layer in enumerate(model.layer_list):
        result = layer.predict(X_test_tmp)
        test_avg, test_feature_new = result[0], result[1]
        y_pred_layer = np.argmax(test_avg, axis=1)
        layer_f1 = f1_score(y_test, y_pred_layer, average='macro') * 100
        test_layer_macro_f1.append(layer_f1)
        if i == best_layer:
            macro_f1 = layer_f1
        X_test_tmp = np.concatenate([X_test_raw, test_feature_new], axis=1)

    if macro_f1 is None:  # fallback if best_layer somehow out of range
        macro_f1 = test_layer_macro_f1[-1] if test_layer_macro_f1 else 0.0

    return {
        'run_id': run_id,
        'accuracy': test_acc,
        'macro_f1': macro_f1,
        'best_layer': best_layer,
        'training_time': training_time,
        'early_stop_training_time': early_stop_training_time,
        'testing_time': testing_time,
        'best_layer_testing_time': best_layer_testing_time,
        'train_layer_accuracy': str(model.val_acc_list),
        'test_layer_accuracy': str(test_acc_list),
        'test_layer_macro_f1': str(test_layer_macro_f1),
        'train_v_info': str(model.v_info_dict['v_info']),
        'test_v_info': str(test_v_info_dict['v_info']),
        'layer_avg_ensemble_size': str(layer_avg_sizes),
    }


def run_experiments_on_dataset(dataset_name):
    """Run multiple experiments on a single dataset."""
    results = []

    print(f"\n{'='*60}")
    print(f"Running experiments on {dataset_name}")
    print(f"{'='*60}")

    for run_id in range(NUM_RUNS):
        print(f"\nRun {run_id + 1}/{NUM_RUNS}")
        random_state = run_id  # run_id as random seed

        result = run_experiment(dataset_name, run_id, random_state)
        results.append(result)

        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Macro-F1: {result['macro_f1']:.2f}%")
        print(f"  Best Layer: {result['best_layer']}")
        print(f"  Training Time (total): {result['training_time']:.2f}s")
        print(f"  Training Time (early-stop): {result['early_stop_training_time']:.2f}s")
        print(f"  Testing Time: {result['testing_time']:.2f}s")
        print(f"  Layer Avg Ensemble Sizes: {result['layer_avg_ensemble_size']}")

    return pd.DataFrame(results)


def main():
    """Main function to run all experiments."""
    # Create results directory
    results_dir = os.path.join(project_root, 'result', 'DF_Vinfo')
    os.makedirs(results_dir, exist_ok=True)

    # Setup logging to file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(results_dir, f'log_{timestamp}.txt')

    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    print(f"Experiment started at {timestamp}")
    print(f"Log file: {log_file}\n")

    # Track which datasets succeed
    successful_datasets = []

    # Run experiments on each dataset
    for dataset_name in DATASETS:
        try:
            df_results = run_experiments_on_dataset(dataset_name)

            # Save results to CSV
            output_file = os.path.join(results_dir, generate_result_filename(MODEL_NAME, dataset_name, VINFO_CONFIG))
            df_results.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")

            # Print summary statistics
            print(f"\nSummary for {dataset_name}:")
            print(f"  Mean Accuracy: {df_results['accuracy'].mean():.2f}% ± {df_results['accuracy'].std():.2f}%")
            print(f"  Mean Macro-F1: {df_results['macro_f1'].mean():.2f}% ± {df_results['macro_f1'].std():.2f}%")
            print(f"  Mean Best Layer: {df_results['best_layer'].mean():.2f}")
            print(f"  Mean Training Time: {df_results['training_time'].mean():.2f}s")
            print(f"  Mean Testing Time: {df_results['testing_time'].mean():.2f}s")

            successful_datasets.append(dataset_name)

        except Exception as e:
            print(f"\nError running experiments on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Successful datasets: {successful_datasets}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
