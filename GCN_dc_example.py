import os
import logging
import warnings

import numpy as np
import torch

# 屏蔽干扰信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import deepchem as dc
import numpy as np

# ===================== 1. 准备示例数据（你可替换为自己的train_dataset） =====================
# 以MUV数据集为例（分类任务），也可替换为你的数据集
tasks, datasets, transformers = dc.molnet.load_muv(
    featurizer='GraphConv',  # 必须用GraphConv featurizer，匹配模型输入
    splitter='random'
)
train_dataset, valid_dataset, test_dataset = datasets
n_tasks = len(tasks)  # 任务数（MUV是17个分类任务）
n_classes = 2  # 二分类任务（多数分子性质预测是二分类：有活性/无活性）

# ===================== 2. 正确获取原子特征维度 =====================
# GraphConvFeaturizer生成的特征中，原子特征维度可通过dataset获取
num_features = train_dataset.get_data_shape()[0]  # 替代你的get_atom_features()

# ===================== 3. 正确初始化GraphConvModel =====================
# 核心修正：导入路径+参数名
model = dc.models.GraphConvModel(
    n_tasks=n_tasks,                # 任务数量
    mode='classification',          # 任务类型（classification/regression）
    n_classes=n_classes,            # 分类任务的类别数（二分类填2）
    n_features=num_features,        # 原子特征维度（替代number_input_features）
    graph_conv_layers=[64, 64],     # 图卷积层的维度（对应你写的64）
    dropout=0.2,                    # 可选：防止过拟合
    learning_rate=0.001             # 可选：学习率
)

# ===================== 4. 训练模型 =====================
model.fit(train_dataset, nb_epoch=50)

# ===================== 5. 验证模型（可选） =====================
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")
train_score = model.evaluate(train_dataset, [metric], transformers)
valid_score = model.evaluate(valid_dataset, [metric], transformers)
print("训练集AUC：", train_score)
print("验证集AUC：", valid_score)
