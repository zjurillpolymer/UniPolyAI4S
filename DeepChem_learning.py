import os
import logging
import warnings

# 屏蔽干扰信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import deepchem as dc
import torch
from deepchem.models.torch_models import GCNModel
from deepchem.feat import MolGraphConvFeaturizer # 必须导入

print(f"DeepChem version: {dc.__version__}")
print(f"PyTorch backend status: {'Ready' if torch.cuda.is_available() else 'CPU Mode'}")

# --- 关键修改：使用实例对象 ---
feat = MolGraphConvFeaturizer()
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=feat)
train_dataset, valid_dataset, test_dataset = datasets

# 初始化 GCN 模型 (PyTorch 实现)
model = GCNModel(
    n_tasks=1,
    mode='regression',
    dropout=0.2,
    learning_rate=0.001
)

print("Training...")
model.fit(train_dataset, nb_epoch=50) # 先跑50轮看看效果

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print("Training set score:", model.evaluate(train_dataset, [metric], transformers))
print("Test set score:", model.evaluate(test_dataset, [metric], transformers))





solubilities = model.predict_on_batch(test_dataset.X[:10])
for molecule, solubility, test_solubility in zip(test_dataset.ids, solubilities, test_dataset.y):
    print(solubility, test_solubility, molecule)

