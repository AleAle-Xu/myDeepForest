import numpy as np
import time
import geatpy as ea
import matplotlib.pyplot as plt
import os

def calculate_margin(scores_matrix, true_labels):
    """
    计算每个样本真实类别上的得分减去剩余类别中的最大得分。

    Args:
        scores_matrix (np.ndarray): 每个样本在各类上的得分组成的矩阵，形状为 (样本数, 类别数)。
        true_labels (np.ndarray): 样本所属的真实类别，形状为 (样本数,)。

    Returns:
        np.ndarray: 每个样本真实类别上的得分减去剩余类别中的最大得分，形状为 (样本数,)。
    """
    true_labels = true_labels.astype(int)
    num_samples, num_classes = scores_matrix.shape
    other_classes = np.arange(num_classes)[:, np.newaxis] != true_labels
    other_classes = other_classes.T
    # print(other_classes.shape, scores_matrix.shape, true_labels.shape)
    max_other_scores = np.max(np.where(other_classes, scores_matrix, -np.inf), axis=1)
    margins = scores_matrix[np.arange(num_samples), true_labels] - max_other_scores

    return margins

def predict_proba_from_treelist(learner_list, X, num_classes): #generate_prob_vector
    n_samples = X.shape[0]
    output = np.zeros((n_samples,num_classes))
    for i in range(len(learner_list)):
        tree = learner_list[i]
        if num_classes==1:
            pred = tree.predict(X)
            pred = pred.reshape(-1,1)
        else:
            pred = tree.predict_proba(X)
        output = output+pred*1.0/len(learner_list)
    return output

# class SelectiveEnsemble(ea.Problem):
#     def __init__(self, n_selected_estimators):
#         self.n_selected_estimators = n_selected_estimators  # number of selected trees
#
#
#     def train(self, train_data, train_label):
#
#         return []
#
#
#     def margin(self, data, label):
#
#         return []

def plot_margin_histograms(margin, unlabeled_margin, layer_id, save_dir='./plots'):
    """
    分别绘制有标记样本和无标记样本的 margin 频率直方图。

    参数：
    - margin: ndarray，有标记样本的 margin
    - unlabeled_margin: ndarray，无标记样本的 margin
    - layer_id: int，层编号（用于保存图像）
    - save_dir: str，保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    bins = np.linspace(-1, 1, 20)

    # 有标记样本直方图
    plt.figure(figsize=(6, 4))
    plt.hist(margin, bins=bins, color='blue', alpha=0.7, density=True)
    plt.xlabel('Margin')
    plt.ylabel('Frequency')
    plt.title(f'Labeled Margin Distribution (Layer {layer_id})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'labeled_margin_layer{layer_id}.png'))
    plt.close()

    # 无标记样本直方图
    plt.figure(figsize=(6, 4))
    plt.hist(unlabeled_margin, bins=bins, color='orange', alpha=0.7, density=True)
    plt.xlabel('Margin')
    plt.ylabel('Frequency')
    plt.title(f'Unlabeled Margin Distribution (Layer {layer_id})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'unlabeled_margin_layer{layer_id}.png'))
    plt.close()


def get_dir_in_root(foldername):
    '''
    获取根目录下的某个文件夹的绝对路径。
    foldername：文件夹名字。
    '''
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 向上追溯到项目根目录(SSDF)
    project_root = os.path.dirname(os.path.dirname(current_file_path))

    # 组合出目录路径
    result_dir = os.path.join(project_root, foldername)

    # 如果目录不存在，则创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir



from collections import Counter


def majority_consistent_indices(unlabeled_prob: np.ndarray, num_forests: int):
    """
    判断哪些未标记样本在至少 2/3 的森林中预测结果一致，并返回这些样本索引及其预测标签。

    Args:
        unlabeled_prob (np.ndarray): shape = (num_forests, num_unlabeled, num_classes)
        num_forests (int): 当前层的森林数量

    Returns:
        consistent_indices (List[int]): 样本索引，满足一致性条件
    """
    num_unlabeled = unlabeled_prob.shape[1]
    consistent_indices = []

    min_agree = max(2, int(np.ceil(2 * num_forests / 3)))  # 至少 2/3 一致，至少为2

    for i in range(num_unlabeled):
        preds = np.argmax(unlabeled_prob[:, i, :], axis=1)  # 每棵森林对第 i 个样本的预测
        pred_counts = Counter(preds)
        most_common_label, count = pred_counts.most_common(1)[0]

        if count >= min_agree:
            consistent_indices.append(i)

    return consistent_indices



def compute_accuracy(label, predict):
    if len(predict.shape) > 1:
        test = np.argmax(predict, axis=1)  #if the array shape is (num,typenum),get the final predict result
    else:
        test = predict
    test_copy = test.astype("int")
    label_copy = label.astype("int")
    acc = np.sum(test_copy == label_copy) * 1.0 / len(label_copy) * 100
    return acc


def mse_consistency(proba1, proba2):
    """
    计算两个概率分布在所有样本上的均方误差
    :param proba1: np.ndarray, shape=(n_samples, n_classes)
    :param proba2: np.ndarray, shape=(n_samples, n_classes)
    :return: float
    """
    return np.mean((proba1 - proba2) ** 2)