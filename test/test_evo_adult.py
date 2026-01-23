"""
Test CascadeForestEvo on Adult dataset.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deepforest.CascadeForestVinfo import CascadeForestVinfo


def main():
    # Load Adult dataset
    dataset_path = os.path.join(project_root, 'dataset', 'Adult.csv')
    df = pd.read_csv(dataset_path)
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    print(f"Dataset: Adult")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Create model (original params as requested)
    num_classes = len(np.unique(y))
    model = CascadeForestVinfo(
        num_estimator=100,      # 100 trees per forest
        num_forests=4,          # 4 forests per layer
        num_classes=num_classes,
        max_layer=100,
        max_depth=10,
        n_fold=3,
        tolerance=3,
        pop_size=100,           # EA population size
        max_gen=100,            # EA generations
        target_size=50          # select 50 trees from 100
    )
    
    # Train
    print("\n=== Training ===")
    best_layer = model.train(X_train, y_train)
    
    # Test
    print("\n=== Testing ===")
    best_acc, test_acc_list, test_v_info_dict = model.test(X_test, y_test)
    
    print(f"\n=== Results ===")
    print(f"Best layer: {best_layer}")
    print(f"Best layer test accuracy: {best_acc:.2f}%")
    print(f"Train V-info per layer: {model.v_info_dict['v_info']}")
    print(f"Test V-info per layer: {test_v_info_dict['v_info']}")


if __name__ == '__main__':
    main()
