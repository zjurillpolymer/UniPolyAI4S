import os
import logging
import warnings
import numpy as np
import torch
from torch import nn

# 屏蔽干扰信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 设置PyTorch设备和数据类型
device = torch.device("cpu")  # 先使用CPU避免GPU兼容问题
torch.set_default_dtype(torch.float32)

import deepchem as dc

# pytorch_model=torch.nn.Sequential(
#     torch.nn.Linear(1024,1000),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(1000,12),
# )
# tasks, datasets, transformers = dc.molnet.load_tox21(
#     featurizer='ECFP',
#     splitter='random'  # 显式指定分割器
# )
# train_dataset, valid_dataset, test_dataset = datasets
# model = dc.models.TorchModel(pytorch_model, dc.models.losses.L2Loss())
# model.fit(train_dataset, nb_epoch=50)
# metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
#
# print('training set score:', model.evaluate(train_dataset, [metric]))
# print('test set score:', model.evaluate(test_dataset, [metric]))


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.dense1=torch.nn.Linear(1024, 1000)
        self.dense2=torch.nn.Linear(1000,1)

    def forward(self,inputs):
        y=torch.nn.functional.relu(self.dense1(inputs))
        y=torch.nn.functional.dropout(y,p=0.5,training=self.training)
        logits=self.dense2(y)
        output=torch.sigmoid(logits)
        return output,logits
torch_model = ClassificationModel()
output_types = ['prediction', 'loss']
model = dc.models.TorchModel(torch_model, dc.models.losses.SigmoidCrossEntropy(), output_types=output_types)


tasks, datasets, transformers = dc.molnet.load_bace_classification(feturizer='ECFP', splitter='scaffold')
train_dataset, valid_dataset, test_dataset = datasets
model.fit(train_dataset, nb_epoch=100)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
print('training set score:', model.evaluate(train_dataset, [metric]))
print('test set score:', model.evaluate(test_dataset, [metric]))

