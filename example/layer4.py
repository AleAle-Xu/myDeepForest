"""
without update_data
"""
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import time
import geatpy as ea
from .utils import *

class MyProblem(ea.Problem):
    def __init__(self, M=3, X_proba_list=[], y=None, unlabeled_proba_list=None, target_size=60):
        name = 'MyProblem'
        maxormins = [-1, -1, 1]  # 第三个目标是最小化与目标数量的差距
        Dim = len(X_proba_list)
        varTypes = [1] * Dim
        lb = [0] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.X_proba_list = X_proba_list
        self.y = y
        self.unlabeled_proba_list = unlabeled_proba_list
        self.target_size = target_size  # 目标选择的树的数量

    def evalVars(self, Vars):
        ObjV = []
        n_learners = len(Vars[0])
        for ind in range(len(Vars)):
            # Vars is a matrix, each line is an individual, with each bit as selector
            s = Vars[ind]
            
            # 计算选择的树的数量
            n_selected = sum(s)
            
            # 计算与目标数量的差距
            size_diff = abs(n_selected - self.target_size)
            
            # 如果选择数量为0，给予惩罚
            if n_selected == 0:
                ObjV.append([-np.inf, np.inf, np.inf])
                continue
            
            # 计算前两个目标
            X_proba_sum = np.zeros_like(self.X_proba_list[0])
            unlabeled_proba_sum = np.zeros_like(self.unlabeled_proba_list[0])
            for i in range(n_learners):
                if s[i] == 1:
                    X_proba_sum += self.X_proba_list[i]
                    unlabeled_proba_sum += self.unlabeled_proba_list[i]

            X_proba = X_proba_sum * 1.0 / n_selected
            unlabeled_proba = unlabeled_proba_sum * 1.0 / n_selected

            # 第一个目标：有标记数据的间隔均值
            margin = calculate_margin(X_proba, self.y)
            mean_margin = np.mean(margin)

            # 第二个目标：无标记数据间隔标准差的负值
            unlabeled_pred = np.argmax(unlabeled_proba, axis=1)
            unlabeled_margin = calculate_margin(unlabeled_proba, unlabeled_pred)
            std_unlabeled_margin = -np.std(unlabeled_margin, ddof=1)

            # 第三个目标：与目标数量的差距
            ObjV.append([mean_margin, std_unlabeled_margin, size_diff])

        ObjV = np.vstack(ObjV)
        return ObjV

class Layer4:
    def __init__(self, num_forests, n_estimators, num_classes,
                 n_fold, layer_index, max_depth=100, min_samples_leaf=1, target_size=60):
        self.num_forests = num_forests  # number of forests
        self.n_estimators = n_estimators  # number of trees in each forest
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_fold = n_fold
        self.layer_index = layer_index
        self.forest_list = []
        self.target_size = target_size


    def generate_base_learners(self, X_train, y_train, forest_index):
        base_learner_list = []
        
        # 将num_estimators平均分配到三个深度
        trees_per_depth = self.n_estimators // 3
        
        # 定义不同深度的树的数量
        depth_counts = {
            2: trees_per_depth,  # 深度为2的树
            4: trees_per_depth,  # 深度为3的树
            6: trees_per_depth   # 深度为4的树
        }
        
        # 为每个深度创建指定数量的树
        for depth, count in depth_counts.items():
            # 根据森林编号选择使用RF还是CRF
            if forest_index % 2 == 0:
                # 使用普通随机森林
                forest = RandomForestClassifier(
                    n_estimators=count,
                    max_depth=depth,
                )
            else:
                # 使用完全随机森林
                forest = ExtraTreesClassifier(
                    n_estimators=count,
                    max_depth=depth,
                    max_features=1,
                )
            forest.fit(X_train, y_train)
            
            # 将森林中的树添加到列表中
            for tree in forest.estimators_:
                base_learner_list.append(tree)
        
        return base_learner_list

    def select_estimators(self, tree_list, X, y, unlabeled_X):
        selected_tree_list = []

        # use geatpy to generate pop and obtain selected_idx
        M=3  # 修改为3个目标
        X_proba_list = []
        unlabeled_proba_list = []
        for i in range(len(tree_list)):
            tree = tree_list[i]
            X_proba_list.append(tree.predict_proba(X))
            unlabeled_proba_list.append(tree.predict_proba(unlabeled_X))

        problem = MyProblem(M=M,X_proba_list=X_proba_list,y=y,unlabeled_proba_list=unlabeled_proba_list,target_size= self.target_size)
        # algorithm = ea.moea_NSGA2_templet(problem,
        #                                   ea.Population(Encoding='BG', NIND=100),
        #                                   MAXGEN=100,
        #                                   logTras=0)

        # 其他可选的 geatpy 多目标优化算法
        # 1. MOEA/D: 基于分解的多目标进化算法
        # algorithm = ea.moea_MOEAD_templet(problem,
        #                                   ea.Population(Encoding='BG', NIND=100),
        #                                   MAXGEN=100,
        #                                   logTras=0)

        # # 2. NSGA-III: NSGA-II 的改进版，适用于更多目标
        algorithm = ea.moea_NSGA3_templet(problem,
                                          ea.Population(Encoding='BG', NIND=100),
                                          MAXGEN=100,
                                          logTras=0)

        # 3. awGA: 自适应权重遗传算法
        # algorithm = ea.moea_awGA_templet(problem,
        #                                   ea.Population(Encoding='BG', NIND=100),
        #                                   MAXGEN=100,
        #                                   logTras=0)

        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
        pop = res['Vars']
        popv = res['ObjV']
        
        # 直接使用第三个目标（与目标数量的差距）选择解
        best_idx = np.argmin(popv[:, 2])  # 选择差距最小的解
        selected_idx = pop[best_idx]
        
        # 输出选择信息
        print(f"选择了 {sum(selected_idx)} 个学习器")
    

        # 构建选中的树列表
        for i, flag in enumerate(selected_idx):
            if flag == 1:
                selected_tree_list.append(tree_list[i])
                
        return selected_tree_list

    def fit(self, X, y, unlabeled_X):
        # If X, y and unlabeled_X are updated during each layer, then update in cascade1.py before call layer.fit

        val_prob = np.zeros((self.num_forests, X.shape[0], self.num_classes), dtype=np.float64)
        unlabeled_prob = np.zeros((self.num_forests, unlabeled_X.shape[0], self.num_classes), dtype=np.float64)
        val_prob_before = np.zeros((self.num_forests, X.shape[0], self.num_classes), dtype=np.float64)
        unlabeled_prob_before = np.zeros((self.num_forests, unlabeled_X.shape[0], self.num_classes), dtype=np.float64)
        n_dim = X.shape[1]

        for forest_index in range(self.num_forests):
            num_classes = int(np.max(y)+1)
            val_prob_forest = np.zeros((X.shape[0], num_classes))
            unlabeled_prob_forest = np.zeros((unlabeled_X.shape[0], num_classes))
            val_prob_forest_before = np.zeros((X.shape[0], num_classes))
            unlabeled_prob_forest_before = np.zeros((unlabeled_X.shape[0], num_classes))

            #tempk = KFold(n_splits=self.n_fold, shuffle=True)
            tempk = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
            kf = []
            for i, j in tempk.split(X,y):
                kf.append([i, j])

            # tic = time.clock()
            # if forest_index % 2 == 0:
            # print('rf')
            kfold = 0
            kfold_list = []
            for train_index, val_index in kf:
                kfold += 1

                X_train = X[train_index, :]
                y_train = y[train_index]
                # clf.fit(X_train, y_train)
                base_learner_list = self.generate_base_learners(X_train, y_train, forest_index)

                # Calculate oof prediction before selection
                tmp_val_prob_before = predict_proba_from_treelist(base_learner_list, X[val_index, :], self.num_classes)
                tmp_unlabeled_prob_before = predict_proba_from_treelist(base_learner_list, unlabeled_X, self.num_classes)
                val_prob_forest_before[val_index, :] = tmp_val_prob_before
                unlabeled_prob_forest_before += tmp_unlabeled_prob_before * 1.0 / self.n_fold

                # call select_estimators
                print(f"kfold:{kfold}")
                # selected_tree_list = self.select_estimators(base_learner_list, X[val_index, :], y[val_index], unlabeled_X)
                # 测试不选择树，使用伪标记的实验
                selected_tree_list = base_learner_list

                kfold_list.append(selected_tree_list)

                tmp_val_prob = predict_proba_from_treelist(selected_tree_list, X[val_index, :],self.num_classes)
                tmp_unlabeled_prob = predict_proba_from_treelist(selected_tree_list, unlabeled_X,self.num_classes)

                val_prob_forest[val_index, :] = tmp_val_prob
                unlabeled_prob_forest += tmp_unlabeled_prob*1.0/self.n_fold

            self.forest_list.append(kfold_list)

            val_prob[forest_index, :] = val_prob_forest
            unlabeled_prob[forest_index, :] = unlabeled_prob_forest
            val_prob_before[forest_index, :] = val_prob_forest_before
            unlabeled_prob_before[forest_index, :] = unlabeled_prob_forest_before

        val_avg = np.sum(val_prob, axis=0)
        val_avg /= self.num_forests
        unlabeled_avg = np.sum(unlabeled_prob, axis=0)
        unlabeled_avg /= self.num_forests

        val_avg_before = np.sum(val_prob_before, axis=0)
        val_avg_before /= self.num_forests
        unlabeled_avg_before = np.sum(unlabeled_prob_before, axis=0)
        unlabeled_avg_before /= self.num_forests

        val_concatenate = val_prob.transpose((1, 0, 2))
        val_concatenate = val_concatenate.reshape(val_concatenate.shape[0], -1)
        unlabeled_concatenate = unlabeled_prob.transpose((1, 0, 2))
        unlabeled_concatenate = unlabeled_concatenate.reshape(unlabeled_concatenate.shape[0], -1)

        return [val_avg, unlabeled_avg, val_concatenate, unlabeled_concatenate, val_avg_before, unlabeled_avg_before]
