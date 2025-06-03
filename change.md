## 截止6/3的commit

1. deepforest包下是最原始的df代码。修改了一点，把KFold改为了StratifiedKFold，防止某些折出现类别不匹配问题。

2. deepforest包下是为使用eoh而修改的df代码主要改动点如下：

   * utils下新增模型转换方法，用于eoh演化。

     ````python
     def in_model_feature_transform(X_raw, predict_prob):
         num_samples = X_raw.shape[0]
         feature_new = predict_prob.transpose((1, 0, 2))
         feature_new = feature_new.reshape((num_samples, -1))
         feature_new = np.concatenate([X_raw, feature_new], axis=1)
         return feature_new
     ````

     其实做的就是原本的根据每层的预测结果获取新特征，以及与原始特征拼接的过程。

     另外新增一个工具方法，从根目录下获取目录。

     ```python
     def get_dir_in_root(foldername):
         '''
         获取根目录下的某个文件夹的绝对路径。
         foldername：文件夹名字。
         '''
         # 获取当前文件的绝对路径
         current_file_path = os.path.abspath(__file__)
         # 向上追溯到项目根目录
         project_root = os.path.dirname(os.path.dirname(current_file_path))
         # 组合出目录路径
         result_dir = os.path.join(project_root, foldername)
         # 如果目录不存在，则创建
         if not os.path.exists(result_dir):
             os.makedirs(result_dir)
         return result_dir
     ```

     

   * Layer中不在进行新特征生成，而是直接返回该层的预测结果。

     ```python
     # feature_new = val_prob.transpose((1, 0, 2))
     # feature_new = feature_new.reshape((num_samples, -1))
     return [val_avg, val_prob]
     ```

   * gcForest中调用用于演化的方法，进行模型内特征转换。

     ```python
     # train_data = np.concatenate([train_data_raw, val_prob], axis=1)
     train_data = in_model_feature_transform(train_data_raw, val_prob)
     ```

3. test下新增dataset_downloader.py，用于下载数据集。根目录下新增dataset目录，保存原始下载的数据。

4. test下测试了两个数据集，df_Adult和df_Gamma。用的是deepforest_eoh中的df。