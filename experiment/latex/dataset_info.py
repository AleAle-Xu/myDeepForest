"""
Generate LaTeX table showing dataset statistics:
- Samples: number of samples
- Features: feature dimensionality
- Classes: number of classes
- Numbers: per-class sample counts (comma separated)
- Imbalance Ratio: max_class_count / min_class_count
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from deepforest.utils import get_dir_in_root

dataset_dir = get_dir_in_root("dataset")

# All datasets (sorted)
DATASETS = sorted([
    'Adult', 'Arrhythmia', 'BankMarketing', 'Car', 'Covertype',
    'CreditCard', 'Diabetes', 'DryBean', 'Gamma', 'HeartDisease',
    'HTRU2', 'Letter', 'Maternal', 'Mushroom', 'Rice', 'Student', 'Websites'
])

dataset_info = []

for name in DATASETS:
    fpath = os.path.join(dataset_dir, f"{name}.csv")
    if not os.path.exists(fpath):
        print(f"Warning: {fpath} not found, skipping.")
        continue

    df = pd.read_csv(fpath)
    y = df['target'].to_numpy()
    X = df.drop(columns=['target'])

    n_samples = len(y)
    n_features = X.shape[1]
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    class_counts_str = ", ".join(str(c) for c in counts)
    ir = float(max(counts)) / float(min(counts))

    dataset_info.append({
        'Dataset': name,
        'Samples': n_samples,
        'Features': n_features,
        'Classes': n_classes,
        'Numbers': class_counts_str,
        'IR': ir
    })
    print(f"{name}: {n_samples} samples, {n_features} features, {n_classes} classes, IR={ir:.2f}")

# Generate LaTeX table
latex_output = []
latex_output.append(r"\begin{table}[htbp]")
latex_output.append(r"\centering")
latex_output.append(r"\caption{Dataset Statistics}")
latex_output.append(r"\label{tab:datasets}")
latex_output.append(r"\resizebox{\textwidth}{!}{")
latex_output.append(r"\begin{tabular}{llcccr}")
latex_output.append(r"\toprule")
latex_output.append(r"Dataset & Samples & Features & Classes & Numbers (per class) & IR \\")
latex_output.append(r"\midrule")

for info in dataset_info:
    name_tex = info['Dataset'].replace('_', r'\_')
    line = (
        f"{name_tex} & {info['Samples']:,} & {info['Features']} & "
        f"{info['Classes']} & {info['Numbers']} & {info['IR']:.2f} \\\\"
    )
    latex_output.append(line)

latex_output.append(r"\bottomrule")
latex_output.append(r"\end{tabular}")
latex_output.append(r"}")
latex_output.append(r"\end{table}")

result = "\n".join(latex_output)
print("\n" + "=" * 60)
print("LaTeX Table:")
print("=" * 60)
print(result)

# Save to file
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_info.tex")
with open(output_path, "w") as f:
    f.write(result)
print(f"\nSaved to {output_path}")
