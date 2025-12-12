# myDeepForest

这是一个基于 Python 实现的深度森林（gcForest）算法。

## 项目结构

```
myDeepForest/
├── dataset/                # 预处理好的数据集，可直接使用
├── dataset_pre_transform/  # 数据集预处理代码
├── dataset_raw/            # 原始数据集
├── deepforest/             # 深度森林算法源代码
├── deepforest_eoh/         # 用于EOH演化而设计的代码结构（可忽略）
├── test/                   # 测试文件
├── environment.yml         # Conda 环境配置文件
└── README.md               # 项目说明文档
```

- **dataset**: 存放预处理后的数据集，可以直接用于模型训练和测试。
- **dataset_pre_transform**: 包含用于处理 `dataset_raw` 中原始数据的 Python 脚本。
- **dataset_raw**: 存放从网络上下载的原始数据集。
- **deepforest**: 深度森林算法的核心实现，包括 `gcForest`、`Layer` 等模块。
- **test**: 包含项目的测试代码。其中 `dataset_downloader.py` 用于下载原始数据集。

## 快速开始

### 1. 环境配置

本项目使用 Conda 进行环境管理。您可以通过以下命令从 `environment.yml` 文件创建并激活 Conda 环境：

```bash
# 从 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate mydeepforest
```

如果 `environment.yml` 文件存在问题，您可以手动创建一个新的 Conda 环境并安装必要的库：

```bash
conda create -n mydeepforest python=3.8
conda activate mydeepforest
pip install numpy scikit-learn
```

### 2. 运行示例

关于如何在不同数据集上运行深度森林模型的示例代码，请参考 `test/test_dataset` 目录下的 Python 脚本。

例如，您可以在激活 Conda 环境后，直接运行 `test/test_dataset/df_Adult.py` 文件来查看在 Adult 数据集上的训练和评估过程：

```bash
python test/test_dataset/df_Adult.py
```
