"""
Plot layer-wise metrics for VIDF experiments.
Supports plotting:
- train_v_info: V-information during training
- test_layer_accuracy: Test accuracy per layer
- test_layer_macro_f1: Test macro-F1 per layer

The best layer is highlighted with a different color marker.
"""
import os
import sys
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from deepforest.utils import get_dir_in_root

# =====================================================================
# Configuration
# =====================================================================
# Choose which metric to plot: 'train_v_info', 'test_layer_accuracy', 'test_layer_macro_f1'
METRIC = 'test_layer_macro_f1'

# Dataset to plot (None = plot all datasets)
DATASET = None  # e.g., 'Adult', 'DNA', etc. Set to None to plot all

# Output directory: figure/{metric_name}/
# e.g., figure/train_v_info/, figure/test_layer_accuracy/, figure/test_layer_macro_f1/
figure_root = get_dir_in_root("figure")
OUTPUT_DIR = os.path.join(figure_root, METRIC)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# Metric configuration
# =====================================================================
METRIC_CONFIG = {
    'train_v_info': {
        'column': 'train_v_info',
        'ylabel': 'V-information',
        'title_suffix': 'Training V-information',
    },
    'test_layer_accuracy': {
        'column': 'test_layer_accuracy',
        'ylabel': 'Test Accuracy (%)',
        'title_suffix': 'Test Accuracy',
    },
    'test_layer_macro_f1': {
        'column': 'test_layer_macro_f1',
        'ylabel': 'Test Macro-F1 (%)',
        'title_suffix': 'Test Macro-F1',
    },
}

if METRIC not in METRIC_CONFIG:
    raise ValueError(f"Invalid METRIC: {METRIC}. Choose from {list(METRIC_CONFIG.keys())}")

config = METRIC_CONFIG[METRIC]

# =====================================================================
# Load results
# =====================================================================
result_dir = get_dir_in_root("result/DF_Vinfo")
csv_files = glob.glob(f"{result_dir}/CascadeForestVinfo_*.csv")

if not csv_files:
    print(f"No result files found in {result_dir}")
    sys.exit(1)

# Extract dataset names
dataset_files = {}
for f in csv_files:
    basename = os.path.basename(f)
    # Format: CascadeForestVinfo_{dataset}_num_estimator...csv
    parts = basename.split('_')
    if len(parts) >= 2:
        dataset_name = parts[1]
        dataset_files[dataset_name] = f

# Filter by DATASET if specified
if DATASET is not None:
    if DATASET not in dataset_files:
        print(f"Dataset {DATASET} not found. Available: {list(dataset_files.keys())}")
        sys.exit(1)
    dataset_files = {DATASET: dataset_files[DATASET]}

print(f"Plotting {METRIC} for {len(dataset_files)} dataset(s): {list(dataset_files.keys())}")

# =====================================================================
# Plot function
# =====================================================================
def plot_metric_for_dataset(dataset_name, csv_path):
    """Plot layer-wise metric for a single dataset."""
    df = pd.read_csv(csv_path)

    if config['column'] not in df.columns or 'best_layer' not in df.columns:
        print(f"Warning: Required columns not found in {csv_path}")
        return

    # Parse metric values (stored as string representation of list)
    metric_values = []
    best_layers = df['best_layer'].values

    for idx, row in df.iterrows():
        try:
            values = ast.literal_eval(row[config['column']])
            metric_values.append(values)
        except:
            print(f"Warning: Failed to parse {config['column']} for run {idx}")
            continue

    if not metric_values:
        print(f"No valid data for {dataset_name}")
        return

    # Convert to numpy array: shape (num_runs, num_layers)
    metric_values = np.array(metric_values)
    num_runs, num_layers = metric_values.shape

    # Compute mean and std across runs
    mean_values = np.mean(metric_values, axis=0)
    std_values = np.std(metric_values, axis=0)

    # Most common best layer
    best_layer_mode = int(np.median(best_layers))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = np.arange(num_layers)

    # Plot mean line with error band
    ax.plot(layers, mean_values, 'o-', linewidth=2, markersize=6,
            label='Mean', color='steelblue', alpha=0.8)
    ax.fill_between(layers, mean_values - std_values, mean_values + std_values,
                     alpha=0.2, color='steelblue')

    # Highlight best layer
    if 0 <= best_layer_mode < num_layers:
        ax.plot(best_layer_mode, mean_values[best_layer_mode], 'r*',
                markersize=20, label=f'Best Layer ({best_layer_mode})', zorder=10)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.set_title(f"{dataset_name} - {config['title_suffix']}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(layers)

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{METRIC}.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

# =====================================================================
# Main
# =====================================================================
for dataset_name, csv_path in sorted(dataset_files.items()):
    plot_metric_for_dataset(dataset_name, csv_path)

print(f"\nAll plots saved to {OUTPUT_DIR}")
