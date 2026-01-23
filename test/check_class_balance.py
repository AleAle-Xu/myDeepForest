import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Dataset list
DATASETS = ['Gamma', 'DryBean', 'CredictCard', 'BankMarketing', 'Adult', 'Diabetes']


def check_class_balance(dataset_name):
    """Check class balance for a dataset."""
    dataset_dir = os.path.join(project_root, 'dataset')
    file_path = os.path.join(dataset_dir, f'{dataset_name}.csv')
    
    df = pd.read_csv(file_path)
    y = df['target'].values
    
    # Count samples per class
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {total}")
    print(f"Number of classes: {len(unique)}")
    print(f"{'-'*50}")
    print(f"{'Class':<10} {'Count':<10} {'Percentage':<10}")
    print(f"{'-'*50}")
    
    for cls, cnt in zip(unique, counts):
        pct = cnt / total * 100
        print(f"{cls:<10} {cnt:<10} {pct:.2f}%")
    
    # Imbalance ratio (max/min)
    imbalance_ratio = max(counts) / min(counts)
    print(f"{'-'*50}")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")


def main():
    """Check class balance for all datasets and save to file."""
    output_file = os.path.join(project_root, 'test', 'class_balance_report.txt')
    
    with open(output_file, 'w') as f:
        for dataset_name in DATASETS:
            try:
                result = get_class_balance_str(dataset_name)
                print(result)
                f.write(result + '\n')
            except Exception as e:
                msg = f"\nError checking {dataset_name}: {e}"
                print(msg)
                f.write(msg + '\n')
    
    print(f"\nReport saved to: {output_file}")


def get_class_balance_str(dataset_name):
    """Get class balance info as string."""
    dataset_dir = os.path.join(project_root, 'dataset')
    file_path = os.path.join(dataset_dir, f'{dataset_name}.csv')
    
    df = pd.read_csv(file_path)
    y = df['target'].values
    
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Total samples: {total}")
    lines.append(f"Number of classes: {len(unique)}")
    lines.append(f"{'-'*50}")
    lines.append(f"{'Class':<10} {'Count':<10} {'Percentage':<10}")
    lines.append(f"{'-'*50}")
    
    for cls, cnt in zip(unique, counts):
        pct = cnt / total * 100
        lines.append(f"{cls:<10} {cnt:<10} {pct:.2f}%")
    
    imbalance_ratio = max(counts) / min(counts)
    lines.append(f"{'-'*50}")
    lines.append(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
