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

from deepforest.gcForest import gcForest

# Dataset list - all datasets that can be run
DATASETS = [
    'Gamma', 'DryBean', 'CreditCard', 'BankMarketing', 'Adult', 'Diabetes',
    'HTRU2', 'Rice', 'Mushroom', 'Websites', 'Letter', 'Car',
    'Maternal', 'Student', 'HeartDisease', 'Covertype'
]

DATASETS = ["Adult", "BankMarketing", "Diabetes", "Gamma", "Student", 
            "Websites",'DNA','Pendigits','Satimage','Segment','Vehicle']

# gcForest configuration
GCFOREST_CONFIG = {
    'num_estimator': 50,
    'num_forests': 4,
    'max_layer': 10,
    'max_depth': 10,
    'n_fold': 3,
    'tolerance': 3
}

# Experiment configuration
NUM_RUNS = 10
TEST_SIZE = 0.3

# Model class name
MODEL_NAME = 'gcForest'


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


def run_experiment(dataset_name, run_id, random_state):
    """Run a single experiment on a dataset."""
    X, y, num_classes = load_dataset(dataset_name)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    # Create model
    model = gcForest(
        num_estimator=GCFOREST_CONFIG['num_estimator'],
        num_forests=GCFOREST_CONFIG['num_forests'],
        num_classes=num_classes,
        max_layer=GCFOREST_CONFIG['max_layer'],
        max_depth=GCFOREST_CONFIG['max_depth'],
        n_fold=GCFOREST_CONFIG['n_fold'],
        tolerance=GCFOREST_CONFIG['tolerance']
    )

    # Train
    train_start = time.time()
    val_p, val_acc, best_layer_index = model.train(X_train, y_train)
    training_time = time.time() - train_start

    # Test
    test_start = time.time()
    test_p, test_acc, best_layer, test_v_info_dict = model.predict(X_test, y_test)
    testing_time = time.time() - test_start

    # Get metrics at best layer
    accuracy = test_acc[best_layer]

    # Compute per-layer macro-F1 on test set
    test_layer_macro_f1 = []
    for prob in test_p:
        y_pred_layer = np.argmax(prob, axis=1)
        test_layer_macro_f1.append(f1_score(y_test, y_pred_layer, average='macro') * 100)

    # Macro-F1 at best layer
    macro_f1 = test_layer_macro_f1[best_layer]

    return {
        'run_id': run_id,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'best_layer': best_layer,
        'training_time': training_time,
        'testing_time': testing_time,
        'train_layer_accuracy': str(model.val_acc_list),
        'test_layer_accuracy': str(test_acc),
        'test_layer_macro_f1': str(test_layer_macro_f1),
        'train_v_info': str(model.v_info_dict['v_info']),
        'test_v_info': str(test_v_info_dict['v_info']),
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

    return pd.DataFrame(results)


def main():
    """Main function to run all experiments."""
    # Create results directory
    results_dir = os.path.join(project_root, 'result_10', 'DF')
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
            output_file = os.path.join(results_dir, generate_result_filename(MODEL_NAME, dataset_name, GCFOREST_CONFIG))
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
