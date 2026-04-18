"""
Plot all three layer-wise metrics for VIDF experiments in one figure.
Plots train_v_info, test_layer_accuracy, and test_layer_macro_f1 as three lines.

Uses only the first run, plots up to best_layer+3 (max 10 layers).
The best layer is highlighted with a marker.
V-info uses left y-axis, accuracy and F1 use right y-axis.
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
# Dataset to plot (None = plot all datasets)
DATASET = None  # e.g., 'Adult', 'DNA', etc. Set to None to plot all

# Output directory: figure/metrics_all/
figure_root = get_dir_in_root("figure")
OUTPUT_DIR = os.path.join(figure_root, "metrics_all")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# Metric configuration
# =====================================================================
METRICS = {
    'train_v_info': {
        'column': 'train_v_info',
        'label': 'Train V-info',
        'color': 'steelblue',
    },
    'test_layer_accuracy': {
        'column': 'test_layer_accuracy',
        'label': 'Test Accuracy (%)',
        'color': 'forestgreen',
    },
    'test_layer_macro_f1': {
        'column': 'test_layer_macro_f1',
        'label': 'Test Macro-F1 (%)',
        'color': 'darkorange',
    },
}

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

print(f"Plotting all metrics for {len(dataset_files)} dataset(s): {list(dataset_files.keys())}")

# =====================================================================
# Plot function
# =====================================================================
def plot_all_metrics_for_dataset(dataset_name, csv_path):
    """Plot all three metrics for a single dataset (first run only)."""
    df = pd.read_csv(csv_path)

    # Use only the first run
    first_row = df.iloc[0]
    best_layer = int(first_row['best_layer'])

    # Parse all three metrics
    metric_data = {}
    for metric_name, metric_config in METRICS.items():
        column = metric_config['column']
        if column not in df.columns:
            print(f"Warning: Column {column} not found in {csv_path}")
            return

        try:
            values = ast.literal_eval(first_row[column])
            metric_data[metric_name] = values
        except:
            print(f"Warning: Failed to parse {column} for {dataset_name}")
            return

    # Determine how many layers to plot: best_layer + 3, max 9 (0-indexed)
    num_layers = len(metric_data['train_v_info'])
    max_layer_to_plot = min(best_layer + 3, num_layers - 1)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    layers = np.arange(max_layer_to_plot + 1)

    for idx, (metric_name, metric_config) in enumerate(METRICS.items()):
        ax = axes[idx]
        values = metric_data[metric_name][:max_layer_to_plot + 1]

        # Plot line
        ax.plot(layers, values, 'o-', linewidth=2, markersize=8,
                color=metric_config['color'], alpha=0.8, label=metric_config['label'])

        # Highlight best layer with vertical line
        ax.axvline(x=best_layer, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Best Layer ({best_layer})')

        ax.set_ylabel(metric_config['label'], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        ax.set_xticks(layers)

    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Layer', fontsize=12)

    # Overall title
    fig.suptitle(f"{dataset_name} - Layer-wise Metrics", fontsize=14, fontweight='bold')

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_all_metrics.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

# =====================================================================
# Main
# =====================================================================
for dataset_name, csv_path in sorted(dataset_files.items()):
    plot_all_metrics_for_dataset(dataset_name, csv_path)

print(f"\nAll plots saved to {OUTPUT_DIR}")
