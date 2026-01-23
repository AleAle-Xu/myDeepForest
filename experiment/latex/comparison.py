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

root = get_dir_in_root("result")
# 配置参数
result_dirs = [
    f"{root}/DF_test_vinfo",
    f"{root}/DF_Vinfo",

]



method_name = ["gcForest", "CascadeForestVinfo"]
column_names = ["DF", "DF_Vinfo"]

query = "accuracy"  # 要提取的指标列

# 自动获取所有数据集名称（通过扫描第一个文件夹中的文件）
# 只匹配以方法名开头的文件，避免匹配到其他文件如final_layers_summary.csv
sample_files = glob.glob(f"{result_dirs[0]}/{method_name[0]}_*.csv")
s = set([os.path.basename(f).split('_')[1] for f in sample_files])
# s.remove("DryBean")
# s.remove("Nursery")
# s.remove("Letter")
dataset_list = sorted(list(s))

# 初始化结果存储
results = {dataset: {version: [] for version in column_names} for dataset in dataset_list}
raw_results = {dataset: {version: [] for version in column_names} for dataset in dataset_list}

# 读取所有数据
for id, path in enumerate(result_dirs):
    version_name = column_names[id]
    method = method_name[id]  # 获取对应的算法名称

    for dataset in dataset_list:
        file_pattern = f"{path}/{method}_{dataset}_*.csv"
        matching_files = glob.glob(file_pattern)
        if not matching_files:
            print(f"Warning: No files found for {dataset} in {path}")
            continue

        # 读取第一个匹配的文件（假设每个数据集在每个版本只有一个文件）
        try:
            df = pd.read_csv(matching_files[0])
            if query in df.columns:
                mean = df[query].mean()
                std = df[query].std()
                results[dataset][version_name] = (mean, std)
                raw_results[dataset][version_name] = df[query].values
            else:
                print(f"Warning: {query} column not found in {matching_files[0]}")
        except Exception as e:
            print(f"Error reading {matching_files[0]}: {e}")

# 计算每个算法的平均排名
method_ranks = {version: [] for version in column_names}
for dataset in dataset_list:
    # 获取当前数据集所有算法的指标值
    metric_values = []
    for version in column_names:
        if results[dataset][version]:
            metric_values.append(results[dataset][version][0])
        else:
            metric_values.append(-np.inf)

    # 计算排名（从1开始）
    ranks = np.argsort(np.argsort(-np.array(metric_values))) + 1

    # 记录每个算法的排名
    for i, version in enumerate(column_names):
        if metric_values[i] != -np.inf:
            method_ranks[version].append(ranks[i])

# 计算平均排名
avg_ranks = {version: np.mean(ranks) for version, ranks in method_ranks.items()}

# 生成LaTeX表格
latex_output = []
latex_output.append("\\begin{tabular}{l|" + "c" * len(column_names) + "}")
latex_output.append("\\toprule")

# 表头
header = "Dataset" + "".join([f" & {name}" for name in column_names]) + " \\\\"
latex_output.append(header)
latex_output.append("\\midrule")

# 表格内容
for dataset in dataset_list:
    line = dataset.replace("_", "\\_")  # 处理LaTeX特殊字符

    # 收集当前数据集的所有版本数据
    version_data = []
    for version in column_names:
        if results[dataset][version]:
            version_data.append(results[dataset][version])
        else:
            version_data.append(None)

    # 确定最佳版本和第二佳版本（均值最大和第二大）
    means = [data[0] if data is not None else -np.inf for data in version_data]
    if any(m > -np.inf for m in means):
        # 获取最佳索引
        best_idx = np.argmax(means)
        # 将最佳值设为负无穷大，然后找第二大的
        temp_means = means.copy()
        temp_means[best_idx] = -np.inf
        second_best_idx = np.argmax(temp_means)
    else:
        best_idx = -1
        second_best_idx = -1

    # 生成表格行
    for i, version in enumerate(column_names):
        data = version_data[i]
        if data is None:
            line += " & $\\times$"
        else:
            mean, std = data
            if i == best_idx:
                line += f" & \\textbf{{{mean:.4f}$\\pm${std:.4f}}}"
            # elif i == second_best_idx:
            #     line += f" & \\underline{{{mean:.4f}$\\pm${std:.4f}}}"
            else:
                line += f" & {mean:.4f}$\\pm${std:.4f}"
            
            # 添加显著性检验标记（与SSDF比较）
            # if version != "SSDF":  # 跳过SSDF自身
            #     ssdf_idx = column_names.index("SSDF")
            #     if raw_results[dataset]["SSDF"] is not None and raw_results[dataset][version] is not None:
            #         # 进行配对Wilcoxon符号秩检验
            #         ssdf_scores = np.array(raw_results[dataset]["SSDF"])
            #         current_scores = np.array(raw_results[dataset][version])
                    
            #         # 计算差值（SSDF - 当前算法）
            #         differences = ssdf_scores - current_scores
                    
            #         # 只有当差值不全为0时才进行检验
            #         if not np.all(differences == 0):
            #             try:
            #                 # 进行双侧检验
            #                 statistics, pvalue = wilcoxon(differences, alternative='two-sided')
                            
            #                 if pvalue < 0.05:  # 显著性水平为0.05
            #                     # 判断SSDF是显著优于还是劣于当前算法
            #                     if np.median(differences) > 0:
            #                         line += "$\\bullet$"  # 表示SSDF显著优于当前算法
            #                     else:
            #                         line += "$\\circ$"  # 表示SSDF显著劣于当前算法
            #             except ValueError:
            #                 # 如果所有差值都相同，wilcoxon会抛出异常，此时不添加标记
            #                 pass

    line += " \\\\"
    latex_output.append(line)

# 添加平均排名行
latex_output.append("\\midrule")
rank_line = "Avg. Rank"
for version in column_names:
    rank_line += f" & {avg_ranks[version]:.2f}"
rank_line += " \\\\"
latex_output.append(rank_line)

# 表格结尾
latex_output.append("\\bottomrule")
latex_output.append("\\end{tabular}")

print("\n".join(latex_output))
# # 输出到文件
# with open("ssdf_comparison_table.tex", "w") as f:
#     f.write("\n".join(latex_output))
#
# print("LaTeX table generated successfully: ssdf_comparison_table.tex")
