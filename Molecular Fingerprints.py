import os
import logging
import warnings
import numpy as np
import torch

# 屏蔽干扰信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 设置PyTorch设备和数据类型
device = torch.device("cpu")  # 先使用CPU避免GPU兼容问题
torch.set_default_dtype(torch.float32)

import deepchem as dc

# 加载Tox21数据集
tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer='ECFP',
    splitter='random'  # 显式指定分割器
)
train_dataset, valid_dataset, test_dataset = datasets

# 打印数据集形状
print("训练集特征形状:", train_dataset.X.shape)
print("训练集标签形状:", train_dataset.y.shape)
print("训练集权重形状:", train_dataset.w.shape)

model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_sizes=[1000])
model.fit(train_dataset, nb_epoch=10)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
print('training set score:', model.evaluate(train_dataset, [metric], transformers))
print('test set score:', model.evaluate(test_dataset, [metric], transformers))