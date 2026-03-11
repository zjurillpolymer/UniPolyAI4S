import os
import warnings
import logging

# 屏蔽无关警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


import deepchem as dc
import numpy as np


tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets
# print(test_dataset)
# print(tasks)
# print(type(transformers))

# print(test_dataset.y.shape) # The dataset contains 113 samples
# print(test_dataset.to_dataframe())


##create datasets
X=np.random.random((10,5))
y=np.random.random((10,2))
dataset=dc.data.NumpyDataset(X=X,y=y)
# print(dataset)
print(dataset.to_dataframe())