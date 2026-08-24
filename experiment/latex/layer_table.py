import os
import sys
import numpy as np
import pandas as pd
import glob
import ast

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from deepforest.utils import get_dir_in_root

root = get_dir_in_root("result_10")

# =====================================================================
# Configuration (same as comparison.py)
# =====================================================================
METHOD_CONFIG = [
    ("DF",        "gcForest",            "Deepforest"),
    ("VIDF_wo_es",  "CascadeForestVinfo",   "VIDF w/o es"),
    ("VIDF_wo_vs",  "CascadeForestVinfo",  "VIDF w/o vs"),
    ("DF_Vinfo",  "CascadeForestVinfo",  "VIDF"),
]

dataset_list = ["Adult", "BankMarketing", "Diabetes", "Gamma", "Student", "Websites",
                'DNA','Pendigits','Satimage','Segment','Vehicle']

column_names = [cfg[2] for cfg in METHOD_CONFIG]

# =====================================================================
# Read best_layer results
# =====================================================================
best_layer_results = {dataset: {name: None for name in column_names} for dataset in dataset_list}

for dir_name, prefix, display_name in METHOD_CONFIG:
    path = os.path.join(root, dir_name)
    if not os.path.exists(path):
        print(f"Warning: Result directory {path} does not exist, skipping {display_name}.")
        continue

    for dataset in dataset_list:
        file_pattern = f"{path}/{prefix}_{dataset}_*.csv"
        matching_files = glob.glob(file_pattern)

        if not matching_files:
            matching_files = glob.glob(f"{path}/{prefix}_{dataset}.csv")

        if not matching_files:
            continue

        try:
            df = pd.read_csv(matching_files[0])
            if 'best_layer' in df.columns:
                vals = df['best_layer'].values
                best_layer_results[dataset][display_name] = (vals.mean(), vals.std())
            else:
                print(f"Warning: 'best_layer' column not found in {matching_files[0]}")
        except Exception as e:
            print(f"Error reading {matching_files[0]}: {e}")

# =====================================================================
# Read layer_avg_ensemble_size results
# =====================================================================
ensemble_size_results = {dataset: {name: None for name in column_names} for dataset in dataset_list}

for dir_name, prefix, display_name in METHOD_CONFIG:
    path = os.path.join(root, dir_name)
    if not os.path.exists(path):
        continue

    for dataset in dataset_list:
        file_pattern = f"{path}/{prefix}_{dataset}_*.csv"
        matching_files = glob.glob(file_pattern)

        if not matching_files:
            matching_files = glob.glob(f"{path}/{prefix}_{dataset}.csv")

        if not matching_files:
            continue

        try:
            df = pd.read_csv(matching_files[0])
            if 'layer_avg_ensemble_size' in df.columns:
                # Each row contains a list of ensemble sizes per layer
                # We need to: 1) average across layers for each run, 2) average across runs
                run_averages = []
                for idx, row in df.iterrows():
                    layer_sizes = ast.literal_eval(row['layer_avg_ensemble_size'])
                    run_avg = np.mean(layer_sizes)
                    run_averages.append(run_avg)

                ensemble_size_results[dataset][display_name] = (np.mean(run_averages), np.std(run_averages))
            else:
                # For methods without ensemble selection (Deepforest), use fixed size of 50
                if display_name == "Deepforest":
                    ensemble_size_results[dataset][display_name] = (50.0, 0.0)
        except Exception as e:
            print(f"Error reading {matching_files[0]}: {e}")

# =====================================================================
# Generate LaTeX table for best_layer
# =====================================================================
def generate_best_layer_table():
    latex_output = []
    latex_output.append("\\begin{table}[htbp]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Average best layer across 5 runs}")
    latex_output.append("\\label{tab:best_layer}")
    latex_output.append("\\resizebox{\\textwidth}{!}{")
    latex_output.append("\\begin{tabular}{l|" + "c" * len(column_names) + "}")
    latex_output.append("\\toprule")

    # Header
    header = "Dataset" + "".join([f" & {name}" for name in column_names]) + " \\\\"
    latex_output.append(header)
    latex_output.append("\\midrule")

    # Rows
    for dataset in dataset_list:
        line = dataset.replace("_", "\\_")

        version_data = [best_layer_results[dataset][name] for name in column_names]

        for i, name in enumerate(column_names):
            data = version_data[i]
            if data is None:
                line += " & $\\times$"
                continue

            mean, std = data
            cell = f"{mean:.2f}$\\pm${std:.2f}"
            line += f" & {cell}"

        line += " \\\\"
        latex_output.append(line)

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("}")
    latex_output.append("\\end{table}")

    return "\n".join(latex_output)

# =====================================================================
# Generate LaTeX table for ensemble_size
# =====================================================================
def generate_ensemble_size_table():
    latex_output = []
    latex_output.append("\\begin{table}[htbp]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Average ensemble size per forest (averaged across layers and 5 runs)}")
    latex_output.append("\\label{tab:ensemble_size}")
    latex_output.append("\\resizebox{\\textwidth}{!}{")
    latex_output.append("\\begin{tabular}{l|" + "c" * len(column_names) + "}")
    latex_output.append("\\toprule")

    # Header
    header = "Dataset" + "".join([f" & {name}" for name in column_names]) + " \\\\"
    latex_output.append(header)
    latex_output.append("\\midrule")

    # Rows
    for dataset in dataset_list:
        line = dataset.replace("_", "\\_")

        version_data = [ensemble_size_results[dataset][name] for name in column_names]

        for i, name in enumerate(column_names):
            data = version_data[i]
            if data is None:
                line += " & $\\times$"
                continue

            mean, std = data
            cell = f"{mean:.2f}$\\pm${std:.2f}"
            line += f" & {cell}"

        line += " \\\\"
        latex_output.append(line)

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("}")
    latex_output.append("\\end{table}")

    return "\n".join(latex_output)

# =====================================================================
# Main execution
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Best Layer Table")
    print("=" * 70)
    best_layer_table = generate_best_layer_table()
    print(best_layer_table)

    # Save best layer table
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_layer_table.tex")
    with open(output_path, "w") as f:
        f.write(best_layer_table)
    print(f"\n% Saved to {output_path}")

    print("\n" + "=" * 70)
    print("Ensemble Size Table")
    print("=" * 70)
    ensemble_size_table = generate_ensemble_size_table()
    print(ensemble_size_table)

    # Save ensemble size table
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ensemble_size_table.tex")
    with open(output_path, "w") as f:
        f.write(ensemble_size_table)
    print(f"\n% Saved to {output_path}")
