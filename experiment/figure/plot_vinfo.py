import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset list
DATASETS = ['Gamma', 'DryBean', 'CredictCard', 'BankMarketing', 'Adult', 'Diabetes']

# V-info metric types
METRIC_TYPES = ['v_info', 'hv_empty', 'hv_cond']
METRIC_LABELS = {
    'v_info': 'V-Information',
    'hv_empty': 'H_V(Y|empty)',
    'hv_cond': 'H_V(Y|X)'
}


def load_results(dataset_name):
    """Load experiment results from CSV file."""
    results_dir = os.path.join(project_root, 'experiment', 'results')
    file_path = os.path.join(results_dir, f'{dataset_name}_results.csv')
    return pd.read_csv(file_path)


def parse_dict_column(df, column_name, metric_key):
    """Parse string representation of dict and extract specific metric."""
    def extract_metric(dict_str):
        d = ast.literal_eval(dict_str)
        return d[metric_key]
    
    lists = df[column_name].apply(extract_metric).tolist()
    return np.array(lists)


def plot_vinfo_for_dataset(dataset_name, metric_type, save_dir):
    """Plot train and test v-info curves for a dataset."""
    df = load_results(dataset_name)
    
    # Parse dict columns and extract specific metric
    train_metric = parse_dict_column(df, 'train_v_info_dict', metric_type)
    test_metric = parse_dict_column(df, 'test_v_info_dict', metric_type)
    
    # Calculate mean across all runs
    train_metric_mean = np.mean(train_metric, axis=0)
    test_metric_mean = np.mean(test_metric, axis=0)
    
    # Number of layers
    num_layers = len(train_metric_mean)
    layers = np.arange(num_layers)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, train_metric_mean, 'b-o', label=f'Train {METRIC_LABELS[metric_type]}', markersize=4)
    plt.plot(layers, test_metric_mean, 'r-s', label=f'Test {METRIC_LABELS[metric_type]}', markersize=4)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(METRIC_LABELS[metric_type], fontsize=12)
    plt.title(f'Layer-wise {METRIC_LABELS[metric_type]} on {dataset_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(save_dir, f'{dataset_name}_{metric_type}.pdf')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def main():
    """
    Generate v-info plots for all datasets.
    """

    # 'v_info', 'hv_empty', 'hv_cond'
    metric_type='hv_empty'
    if metric_type not in METRIC_TYPES:
        raise ValueError(f"metric_type must be one of {METRIC_TYPES}")
    
    # Create figures directory
    figures_dir = os.path.join(project_root, 'experiment', 'figures', 'vinfo')
    os.makedirs(figures_dir, exist_ok=True)
    
    for dataset_name in DATASETS:
        try:
            plot_vinfo_for_dataset(dataset_name, metric_type, figures_dir)
        except Exception as e:
            print(f"Error plotting {dataset_name}: {e}")


if __name__ == '__main__':
    main()
