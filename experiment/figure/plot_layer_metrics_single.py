"""
Plot layer-wise metrics for VIDF experiments (single metric per plot).
Supports plotting:
- train_v_info: V-information during training
- test_layer_accuracy: Test accuracy per layer
- test_layer_macro_f1: Test macro-F1 per layer

Uses only the first run, plots up to best_layer+3 (max 10 layers).
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
METRIC = 'test_layer_accuracy'

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
    """Plot layer-wise metric for a single dataset (first run only)."""
    df = pd.read_csv(csv_path)

    if config['column'] not in df.columns or 'best_layer' not in df.columns:
        print(f"Warning: Required columns not found in {csv_path}")
        return

    # Use only the first run
    first_row = df.iloc[0]
    best_layer = int(first_row['best_layer'])

    try:
        metric_values = ast.literal_eval(first_row[config['column']])
    except:
        print(f"Warning: Failed to parse {config['column']} for {dataset_name}")
        return

    # Determine how many layers to plot: best_layer + 3, max 9 (0-indexed)
    max_layer_to_plot = min(best_layer + 3, len(metric_values) - 1)

    # Truncate to layers we want to plot (inclusive)
    metric_values = metric_values[:max_layer_to_plot + 1]
    layers = np.arange(len(metric_values))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line
    ax.plot(layers, metric_values, 'o-', linewidth=2, markersize=8,
            color='steelblue', alpha=0.8)

    # Highlight best layer
    if 0 <= best_layer < len(metric_values):
        ax.plot(best_layer, metric_values[best_layer], 'r*',
                markersize=20, label=f'Best Layer ({best_layer})', zorder=10)

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
