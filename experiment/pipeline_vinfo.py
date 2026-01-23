"""
Pipeline for running experiments with CascadeForestVinfo (evolutionary selective ensemble).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deepforest.CascadeForestVinfo import CascadeForestVinfo

# Dataset list
DATASETS = ['Gamma', 'DryBean', 'CredictCard', 'BankMarketing', 'Adult', 'Diabetes']

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
NUM_RUNS = 10
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


def run_experiment(dataset_name, run_id, random_state):
    """Run a single experiment on a dataset."""
    X, y, num_classes = load_dataset(dataset_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )
    
    # Create and train model
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
    best_layer = model.train(X_train, y_train)
    
    # Test
    test_acc, test_acc_list, test_v_info_dict = model.test(X_test, y_test)
    
    return {
        'run_id': run_id,
        'accuracy': test_acc,
        'best_layer': best_layer,
        'train_layer_accuracy': model.val_acc_list,
        'test_layer_accuracy': test_acc_list,
        'train_v_info_dict': model.v_info_dict,
        'test_v_info_dict': test_v_info_dict
    }


def run_experiments_on_dataset(dataset_name):
    """Run multiple experiments on a single dataset."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running experiments on {dataset_name}")
    print(f"{'='*60}")
    
    for run_id in range(NUM_RUNS):
        print(f"\nRun {run_id + 1}/{NUM_RUNS}")
        random_state = run_id  # Different random state for each run
        
        result = run_experiment(dataset_name, run_id, random_state)
        results.append(result)
        
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Best Layer: {result['best_layer']}")
    
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
    
    # Run experiments on each dataset
    for dataset_name in DATASETS:
        try:
            df_results = run_experiments_on_dataset(dataset_name)
            
            # Save results to CSV
            output_file = os.path.join(results_dir, f'{dataset_name}_results.csv')
            df_results.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            
            # Print summary statistics
            print(f"\nSummary for {dataset_name}:")
            print(f"  Mean Accuracy: {df_results['accuracy'].mean():.2f}% ± {df_results['accuracy'].std():.2f}%")
            print(f"  Mean Best Layer: {df_results['best_layer'].mean():.2f}")
            
        except Exception as e:
            print(f"\nError running experiments on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
