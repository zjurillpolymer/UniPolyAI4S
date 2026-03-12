import os
import logging
import warnings

# 屏蔽干扰信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")



import deepchem as dc



'''所有 MoleculeNet 加载器函数都采用 dc.molnet.load_X 的形式。加载器函数返回一个元组，其中包含 (tasks, datasets, transformers) 的参数。'''
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer="ECFP", splitter="scaffold")

# for method in dir(dc.molnet):
#     if method.startswith('load_'):
#         print(method)

# print(tasks)

train,valid,test=datasets
print(train.X[0])