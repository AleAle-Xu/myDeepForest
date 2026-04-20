"""
Fix best_layer, accuracy, and macro_f1 in experiment result CSVs.

Rules:
- DF_Vinfo: early stopping on train_v_info,        bad_count >= 3 (stops when bad_count reaches 3)
- DF:        early stopping on train_layer_accuracy, bad > 3       (stops when bad reaches 4)

For each row:
1. Recompute best_layer from the correct metric list using the correct early-stopping rule.
2. Fix accuracy = test_layer_accuracy[best_layer]
3. Fix macro_f1  = test_layer_macro_f1[best_layer]
"""

import ast
import os
import re
import glob
import pandas as pd

VINFO_DIR = "/home/xjl/pythoncode/myDeepForest/result/DF_Vinfo"
DF_DIR    = "/home/xjl/pythoncode/myDeepForest/result/DF"
TOLERANCE = 3


def parse_list(s: str) -> list:
    """Parse a stringified Python list, handling np.float64(...) wrappers."""
    cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)
    return ast.literal_eval(cleaned)


def compute_best_layer_vinfo(v_info_list: list) -> int:
    """
    CascadeForestVinfo early stopping (bad_count >= tolerance):
      if v_info > best: update best, reset bad_count
      else:             bad_count += 1
      stop when bad_count >= TOLERANCE
    """
    best_v_info = -float('inf')
    best_layer = 0
    bad_count = 0
    for i, v in enumerate(v_info_list):
        if v > best_v_info:
            best_v_info = v
            best_layer = i
            bad_count = 0
        else:
            bad_count += 1
        if bad_count >= TOLERANCE:
            break
    return best_layer


def compute_best_layer_df(train_acc_list: list) -> int:
    """
    gcForest early stopping (bad >= tolerance):
      if acc > best: update best, reset bad
      else:          bad += 1
      stop when bad >= TOLERANCE (i.e. bad == 3)
    """
    best_acc = 0
    best_layer = 0
    bad = 0
    for i, acc in enumerate(train_acc_list):
        if acc > best_acc:
            best_acc = acc
            best_layer = i
            bad = 0
        else:
            bad += 1
        if bad >= TOLERANCE:
            break
    return best_layer


def fix_csv(filepath: str, mode: str):
    """mode: 'vinfo' or 'df'"""
    df = pd.read_csv(filepath)
    changed_rows = []

    for idx, row in df.iterrows():
        if mode == 'vinfo':
            metric_list = parse_list(row['train_v_info'])
            correct_best = compute_best_layer_vinfo(metric_list)
        else:
            metric_list = parse_list(row['train_layer_accuracy'])
            correct_best = compute_best_layer_df(metric_list)

        test_acc_list = parse_list(row['test_layer_accuracy'])
        test_f1_list  = parse_list(row['test_layer_macro_f1'])

        old_best = int(row['best_layer'])
        old_acc  = row['accuracy']
        old_f1   = row['macro_f1']
        new_acc  = test_acc_list[correct_best]
        new_f1   = test_f1_list[correct_best]

        if correct_best != old_best or abs(new_acc - old_acc) > 1e-9 or abs(new_f1 - old_f1) > 1e-9:
            changed_rows.append({
                'run_id':   row['run_id'],
                'old_best': old_best, 'new_best': correct_best,
                'old_acc':  old_acc,  'new_acc':  new_acc,
                'old_f1':   old_f1,   'new_f1':   new_f1,
            })
            df.at[idx, 'best_layer'] = correct_best
            df.at[idx, 'accuracy']   = new_acc
            df.at[idx, 'macro_f1']   = new_f1

    if changed_rows:
        df.to_csv(filepath, index=False)
        print(f"[FIXED] {os.path.basename(filepath)}")
        for r in changed_rows:
            print(f"  run {r['run_id']}: best_layer {r['old_best']} -> {r['new_best']}, "
                  f"acc {r['old_acc']:.4f} -> {r['new_acc']:.4f}, "
                  f"f1 {r['old_f1']:.4f} -> {r['new_f1']:.4f}")
    else:
        print(f"[OK]    {os.path.basename(filepath)}")


def main():
    print("=== Fixing DF_Vinfo results (metric: train_v_info, bad_count >= 3) ===")
    for fname in sorted(os.listdir(VINFO_DIR)):
        if fname.endswith('.csv'):
            fix_csv(os.path.join(VINFO_DIR, fname), mode='vinfo')

    print("\n=== Fixing DF results (metric: train_layer_accuracy, bad >= 3) ===")
    for fname in sorted(os.listdir(DF_DIR)):
        if fname.endswith('.csv'):
            fix_csv(os.path.join(DF_DIR, fname), mode='df')


if __name__ == '__main__':
    main()
