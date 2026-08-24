import os
import sys
import numpy as np
import pandas as pd
import glob
from scipy.stats import wilcoxon

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from deepforest.utils import get_dir_in_root

root = get_dir_in_root("result_10")

# =====================================================================
# Configuration
# =====================================================================
# Each entry: (result_dir_name, method_file_prefix, display_name)
# method_file_prefix is used to match files: {prefix}_{dataset}_*.csv
METHOD_CONFIG = [
    ("DF_Vinfo",  "CascadeForestVinfo",  "VIDF"),
    ("DF",        "gcForest",            "Deepforest"),
    ("RF",        "RF",                  "Random Forest"),
    ("ExtraTrees","ExtraTrees",          "Extra Trees"),
    ("XGBoost",   "XGBoost",             "XGBoost"),
    ("TabNet",    "TabNet",              "TabNet"),
]


METHOD_CONFIG = [
    
    ("DF",        "gcForest",            "Deepforest"),
    ("VIDF_wo_es",  "CascadeForestVinfo",   "VIDF w/o es"),
    ("VIDF_wo_vs",  "CascadeForestVinfo",  "VIDF w/o vs"),
    ("DF_Vinfo",  "CascadeForestVinfo",  "VIDF"),
]
# Metric to compare (column name in csv files)
query = "accuracy"   # accuracy or "macro_f1"

# Name of our proposed method (display_name in METHOD_CONFIG)
# Significance markers are added to OTHER methods' cells,
# indicating whether VIDF is significantly better (bullet) or worse (circ).
OUR_METHOD_NAME = "VIDF"

# Whether to run significance tests (Wilcoxon signed-rank, alpha=0.05)
ENABLE_SIGNIFICANCE_TEST = True

# Significance level
ALPHA = 0.05

# =====================================================================
# Discover datasets from DF results (reference method)
# =====================================================================
ref_dir = os.path.join(root, "DF")
ref_prefix = "gcForest"
sample_files = glob.glob(f"{ref_dir}/{ref_prefix}_*.csv")
dataset_list = sorted(set(
    os.path.basename(f).split('_')[1]
    for f in sample_files
))

if not dataset_list:
    print(f"Warning: No result files found in {ref_dir}. Run pipeline.py first.")
    dataset_list = []

column_names = [cfg[2] for cfg in METHOD_CONFIG]

dataset_list = ["Adult", "BankMarketing", "Diabetes", "Gamma", "Student", "Websites",'DNA','Pendigits','Satimage','Segment','Vehicle']
# =====================================================================
# Read all results
# =====================================================================
results     = {dataset: {name: None for name in column_names} for dataset in dataset_list}
raw_results = {dataset: {name: None for name in column_names} for dataset in dataset_list}

for dir_name, prefix, display_name in METHOD_CONFIG:
    path = os.path.join(root, dir_name)
    if not os.path.exists(path):
        print(f"Warning: Result directory {path} does not exist, skipping {display_name}.")
        continue

    for dataset in dataset_list:
        file_pattern = f"{path}/{prefix}_{dataset}_*.csv"
        matching_files = glob.glob(file_pattern)

        # Fallback: exact name without wildcard suffix (baseline models)
        if not matching_files:
            matching_files = glob.glob(f"{path}/{prefix}_{dataset}.csv")

        if not matching_files:
            continue

        try:
            df = pd.read_csv(matching_files[0])
            if query in df.columns:
                vals = df[query].values
                results[dataset][display_name]     = (vals.mean(), vals.std())
                raw_results[dataset][display_name] = vals
            else:
                print(f"Warning: '{query}' column not found in {matching_files[0]}")
        except Exception as e:
            print(f"Error reading {matching_files[0]}: {e}")

# =====================================================================
# Significance test helper
# =====================================================================
def significance_marker(our_scores, their_scores):
    """
    Paired Wilcoxon signed-rank test between VIDF and another method.
    Returns:
        '$^{\\bullet}$'  if VIDF is significantly BETTER  (p < ALPHA, VIDF mean > other mean)
        '$^{\\circ}$'    if VIDF is significantly WORSE   (p < ALPHA, VIDF mean < other mean)
        ''               otherwise (not significant, or unequal run counts)
    """
    if our_scores is None or their_scores is None:
        return ""
    if len(our_scores) != len(their_scores):
        return ""
    differences = our_scores - their_scores
    # Wilcoxon requires at least one nonzero difference
    if np.all(differences == 0):
        return ""
    try:
        _, p_value = wilcoxon(differences, alternative='two-sided')
    except ValueError:
        return ""
    if p_value < ALPHA:
        return r" $^{\bullet}$" if np.mean(our_scores) > np.mean(their_scores) else r" $^{\circ}$"
    return ""

# =====================================================================
# Compute average rank per method
# =====================================================================
method_ranks = {name: [] for name in column_names}
for dataset in dataset_list:
    metric_values = [
        results[dataset][name][0] if results[dataset][name] is not None else -np.inf
        for name in column_names
    ]
    ranks = np.argsort(np.argsort(-np.array(metric_values))) + 1
    for i, name in enumerate(column_names):
        if metric_values[i] != -np.inf:
            method_ranks[name].append(ranks[i])

avg_ranks = {
    name: np.mean(r) if r else float('nan')
    for name, r in method_ranks.items()
}

# =====================================================================
# Generate LaTeX table
# =====================================================================
latex_output = []
latex_output.append("\\begin{table}[htbp]")
latex_output.append("\\centering")
latex_output.append(f"\\caption{{Comparison of methods on {query} (\\%)}}")
latex_output.append("\\label{tab:comparison}")
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

    version_data   = [results[dataset][name]     for name in column_names]
    version_raw    = [raw_results[dataset][name] for name in column_names]
    means          = [d[0] if d is not None else -np.inf for d in version_data]

    our_idx        = column_names.index(OUR_METHOD_NAME) if OUR_METHOD_NAME in column_names else -1
    our_scores_arr = version_raw[our_idx] if our_idx >= 0 else None

    # Determine best / second-best indices
    if any(m > -np.inf for m in means):
        best_idx = int(np.argmax(means))
        temp = means.copy()
        temp[best_idx] = -np.inf
        second_best_idx = int(np.argmax(temp))
    else:
        best_idx = second_best_idx = -1

    for i, name in enumerate(column_names):
        data = version_data[i]
        if data is None:
            line += " & $\\times$"
            continue

        mean, std = data

        # Bold best, underline second-best
        if i == best_idx:
            cell = f"\\textbf{{{mean:.4f}$\\pm${std:.4f}}}"
        elif i == second_best_idx:
            cell = f"\\underline{{{mean:.4f}$\\pm${std:.4f}}}"
        else:
            cell = f"{mean:.4f}$\\pm${std:.4f}"

        # Append significance marker on non-VIDF columns
        if ENABLE_SIGNIFICANCE_TEST and name != OUR_METHOD_NAME:
            marker = significance_marker(our_scores_arr, version_raw[i])
            cell += marker

        line += f" & {cell}"

    line += " \\\\"
    latex_output.append(line)

# Average rank row
latex_output.append("\\midrule")
rank_line = "Avg. Rank"
for name in column_names:
    r = avg_ranks[name]
    rank_line += f" & {r:.2f}" if not np.isnan(r) else " & --"
rank_line += " \\\\"
latex_output.append(rank_line)

latex_output.append("\\bottomrule")
latex_output.append("\\end{tabular}")
latex_output.append("}")  # close \resizebox
latex_output.append("\\end{table}")

# Legend note
if ENABLE_SIGNIFICANCE_TEST:
    latex_output.append(
        f"% Significance markers (Wilcoxon, α={ALPHA}): "
        r"$^{\bullet}$ VIDF significantly better; "
        r"$^{\circ}$ VIDF significantly worse"
    )

print("\n".join(latex_output))

# Save to file
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_table.tex")
with open(output_path, "w") as f:
    f.write("\n".join(latex_output))
print(f"\n% Saved to {output_path}")
