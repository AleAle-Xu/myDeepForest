import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deepforest.gcForest import gcForest

# Dataset list
DATASETS = ['Gamma', 'DryBean', 'CredictCard', 'BankMarketing', 'Adult', 'Diabetes']

# gcForest configuration
GCFOREST_CONFIG = {
    'num_estimator': 100,
    'num_forests': 4,
    'max_layer': 10,
    'max_depth': 10,
    'n_fold': 3,
    'tolerance': 3
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
    val_p, val_acc, best_layer_index = model.train(X_train, y_train)
    
    # Test
    test_p, test_acc, best_layer, test_v_info_dict = model.predict(X_test, y_test)
    
    # Get metrics at best layer
    accuracy = test_acc[best_layer]
    
    return {
        'run_id': run_id,
        'accuracy': accuracy,
        'best_layer': best_layer,
        'train_layer_accuracy': model.val_acc_list,
        'test_layer_accuracy': test_acc,
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
    results_dir = os.path.join(project_root, 'experiment', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
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
