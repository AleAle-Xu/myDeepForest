import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset list
DATASETS = ['Gamma', 'DryBean', 'CredictCard', 'BankMarketing', 'Adult', 'Diabetes']


def load_results(dataset_name):
    """Load experiment results from CSV file."""
    results_dir = os.path.join(project_root, 'experiment', 'results')
    file_path = os.path.join(results_dir, f'{dataset_name}_results.csv')
    return pd.read_csv(file_path)


def parse_list_column(df, column_name):
    """Parse string representation of list back to numpy array."""
    import ast
    lists = df[column_name].apply(ast.literal_eval).tolist()
    return np.array(lists)


def plot_accuracy_for_dataset(dataset_name, save_dir):
    """Plot train and test accuracy curves for a dataset."""
    df = load_results(dataset_name)
    
    # Parse list columns
    train_acc = parse_list_column(df, 'train_layer_accuracy')
    test_acc = parse_list_column(df, 'test_layer_accuracy')
    
    # Calculate mean across all runs
    train_acc_mean = np.mean(train_acc, axis=0)
    test_acc_mean = np.mean(test_acc, axis=0)
    
    # Number of layers
    num_layers = len(train_acc_mean)
    layers = np.arange(num_layers)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, train_acc_mean, 'b-o', label='Train Accuracy', markersize=4)
    plt.plot(layers, test_acc_mean, 'r-s', label='Test Accuracy', markersize=4)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Layer-wise Accuracy on {dataset_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(save_dir, f'{dataset_name}_accuracy.pdf')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def main():
    """Generate accuracy plots for all datasets."""
    # Create figures directory
    figures_dir = os.path.join(project_root, 'experiment', 'figures', 'accuracy')
    os.makedirs(figures_dir, exist_ok=True)
    
    for dataset_name in DATASETS:
        try:
            plot_accuracy_for_dataset(dataset_name, figures_dir)
        except Exception as e:
            print(f"Error plotting {dataset_name}: {e}")


if __name__ == '__main__':
    main()
