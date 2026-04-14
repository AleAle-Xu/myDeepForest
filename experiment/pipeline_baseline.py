"""
Baseline pipeline for traditional ML algorithms:
RF (RandomForest), ExtraTrees, XGBoost, TabNet
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Dataset list - same as DF pipeline (datasets that DF can run)
# This will be populated dynamically from DF results, or you can set manually.
# For safety we use the same full list as pipeline.py; skipped datasets from DF
# will be excluded by checking if DF result files exist.
DATASETS = [
    'Gamma', 'DryBean', 'CredictCard', 'BankMarketing', 'Adult', 'Diabetes',
    'HTRU2', 'Rice', 'Mushroom', 'Websites', 'Letter', 'Car',
    'Maternal', 'Student', 'HeartDisease', 'Covertype'
]

# Model names to run
MODELS = ['RF', 'ExtraTrees', 'XGBoost', 'TabNet']

# Model hyperparameters
MODEL_CONFIGS = {
    'RF': {
        'n_estimators': 100,
        'max_depth': 10,
        'n_jobs': -1,
        'random_state': None,   # set per run
    },
    'ExtraTrees': {
        'n_estimators': 100,
        'max_depth': 10,
        'n_jobs': -1,
        'random_state': None,
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': None,
        'n_jobs': -1,
    },
    'TabNet': {
        'n_d': 16,
        'n_a': 16,
        'n_steps': 3,
        'gamma': 1.3,
        'seed': None,  # set per run
        'verbose': 0,
    },
}

# Experiment configuration
NUM_RUNS = 5
TEST_SIZE = 0.3


def load_dataset(dataset_name):
    """Load dataset from csv file."""
    dataset_dir = os.path.join(project_root, 'dataset')
    file_path = os.path.join(dataset_dir, f'{dataset_name}.csv')

    df = pd.read_csv(file_path)
    X = df.drop(columns=['target']).to_numpy()
    y = df['target'].to_numpy()
    num_classes = len(np.unique(y))

    return X, y, num_classes


def create_model(model_name, num_classes, random_state):
    """Create a model instance based on model_name."""
    if model_name == 'RF':
        cfg = MODEL_CONFIGS['RF'].copy()
        cfg['random_state'] = random_state
        return RandomForestClassifier(**cfg)

    elif model_name == 'ExtraTrees':
        cfg = MODEL_CONFIGS['ExtraTrees'].copy()
        cfg['random_state'] = random_state
        return ExtraTreesClassifier(**cfg)

    elif model_name == 'XGBoost':
        cfg = MODEL_CONFIGS['XGBoost'].copy()
        cfg['random_state'] = random_state
        cfg.pop('use_label_encoder', None)  # deprecated in newer XGBoost
        return XGBClassifier(**cfg)

    elif model_name == 'TabNet':
        cfg = MODEL_CONFIGS['TabNet'].copy()
        cfg['seed'] = random_state
        return TabNetClassifier(**cfg)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(dataset_name, model_name, run_id, random_state):
    """Run a single experiment."""
    X, y, num_classes = load_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    model = create_model(model_name, num_classes, random_state)

    # Train
    train_start = time.time()
    if model_name == 'TabNet':
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            patience=20,
            max_epochs=200,
            batch_size=1024,
            virtual_batch_size=128,
        )
    else:
        model.fit(X_train, y_train)
    train_end = time.time()
    training_time = train_end - train_start

    # Test
    test_start = time.time()
    y_pred = model.predict(X_test)
    test_end = time.time()
    testing_time = test_end - test_start

    accuracy = accuracy_score(y_test, y_pred) * 100
    macro_f1 = f1_score(y_test, y_pred, average='macro') * 100

    return {
        'run_id': run_id,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'training_time': training_time,
        'testing_time': testing_time,
    }


def run_experiments_on_dataset(dataset_name, model_name):
    """Run multiple experiments on a single dataset for one model."""
    results = []

    print(f"\n{'='*60}")
    print(f"[{model_name}] Running experiments on {dataset_name}")
    print(f"{'='*60}")

    for run_id in range(NUM_RUNS):
        print(f"\nRun {run_id + 1}/{NUM_RUNS}")
        random_state = run_id  # run_id as random seed

        result = run_experiment(dataset_name, model_name, run_id, random_state)
        results.append(result)

        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Macro-F1: {result['macro_f1']:.2f}%")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print(f"  Testing Time: {result['testing_time']:.2f}s")

    return pd.DataFrame(results)


def get_df_successful_datasets(results_base_dir):
    """Read DF result dir to find which datasets DF successfully ran."""
    df_dir = os.path.join(results_base_dir, 'DF')
    if not os.path.exists(df_dir):
        return None  # DF not run yet, use all datasets
    import glob
    files = glob.glob(os.path.join(df_dir, 'gcForest_*.csv'))
    datasets = set()
    for f in files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 2:
            datasets.add(parts[1])
    return sorted(list(datasets)) if datasets else None


def main():
    """Main function to run all baseline experiments."""
    results_base_dir = os.path.join(project_root, 'result')

    # Try to use only datasets where DF succeeded
    df_datasets = get_df_successful_datasets(results_base_dir)
    if df_datasets:
        datasets_to_run = [d for d in DATASETS if d in df_datasets]
        print(f"Running on datasets where DF succeeded: {datasets_to_run}")
    else:
        datasets_to_run = DATASETS
        print(f"DF results not found, running on all datasets: {datasets_to_run}")

    # Create results directories for each model
    for model_name in MODELS:
        results_dir = os.path.join(results_base_dir, model_name)
        os.makedirs(results_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(results_base_dir, f'baseline_log_{timestamp}.txt')

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
    print(f"Baseline experiment started at {timestamp}")
    print(f"Log file: {log_file}\n")
    print(f"Models: {MODELS}")
    print(f"Datasets: {datasets_to_run}\n")

    for model_name in MODELS:
        results_dir = os.path.join(results_base_dir, model_name)
        successful_datasets = []

        print(f"\n{'#'*60}")
        print(f"# Model: {model_name}")
        print(f"{'#'*60}")

        for dataset_name in datasets_to_run:
            try:
                df_results = run_experiments_on_dataset(dataset_name, model_name)

                # Save results
                output_file = os.path.join(results_dir, f"{model_name}_{dataset_name}.csv")
                df_results.to_csv(output_file, index=False)
                print(f"\nResults saved to {output_file}")

                print(f"\nSummary for {dataset_name} [{model_name}]:")
                print(f"  Mean Accuracy: {df_results['accuracy'].mean():.2f}% ± {df_results['accuracy'].std():.2f}%")
                print(f"  Mean Macro-F1: {df_results['macro_f1'].mean():.2f}% ± {df_results['macro_f1'].std():.2f}%")
                print(f"  Mean Training Time: {df_results['training_time'].mean():.2f}s")
                print(f"  Mean Testing Time: {df_results['testing_time'].mean():.2f}s")

                successful_datasets.append(dataset_name)

            except Exception as e:
                print(f"\nError running {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n[{model_name}] Successful datasets: {successful_datasets}")


if __name__ == '__main__':
    main()
