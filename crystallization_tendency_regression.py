import os
import logging
import warnings

import numpy as np
import torch


# 屏蔽干扰信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# All Imports
import pandas as pd
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
from sklearn.model_selection import train_test_split
IPythonConsole.ipython_useSVG=True



import urllib.request

# 下载文件到本地
url = "https://media.githubusercontent.com/media/ChangwenXu98/TransPolymer/master/data/Xc.csv"
urllib.request.urlretrieve(url, "Xc.csv")




df = pd.read_csv("./Xc.csv")
print("Number of data points", df.shape[0])
df.head() ##432 points in total
print(type(df))



'''划分训练集和测试集'''
train_df,test_df=train_test_split(df,test_size=0.1,random_state=42)
print("Number of data points in train set", train_df.shape[0])
print("Number of data points in test set", test_df.shape[0])

train_df.to_csv("train_Xc.csv",index=False)
test_df.to_csv("test_Xc.csv",index=False)


smiles="*CC(*)C"
mol=Chem.MolFromSmiles(smiles)
Draw.MolToImage(mol, kekulize=False, size=(200, 200))


molgraph_conv_featureizer=MolGraphConvFeaturizer()
X_molgraph_conv=molgraph_conv_featureizer.featurize(train_df["smiles"].values)
y_molgraph_conv=train_df['value'].values
molgraph_conv_dataset=dc.data.NumpyDataset(X_molgraph_conv, y_molgraph_conv)

X_molgraph_conv_test=molgraph_conv_featureizer.featurize(test_df["smiles"].values)
y_molgraph_conv_test=test_df['value'].values
molgraph_conv_dataset_test=dc.data.NumpyDataset(X_molgraph_conv_test, y_molgraph_conv_test)

metric=dc.metrics.Metric(dc.metrics.mean_squared_error)
losses_gcn,val_losses_gcn=[],[]
GCN_MODEL=dc.models.GCNModel(
    mode='regression',
    n_tasks=1,
    batch_size=2,
    learning_rate=0.001,
)





for n in range(20):
    l2_loss_GCN = GCN_MODEL.fit(molgraph_conv_dataset, nb_epoch=1)
    losses_gcn.append(l2_loss_GCN)
    val_loss_GCN = GCN_MODEL.evaluate(molgraph_conv_dataset_test, [metric])
    val_losses_gcn.append(val_loss_GCN)


# setting up the featurizer
MAT_featurizer = dc.feat.MATFeaturizer()

# featurizing and preparing the dataset
MAT_data_loader = dc.data.CSVLoader(tasks=['value'], feature_field='smiles', featurizer=MAT_featurizer)
MAT_dataset = MAT_data_loader.create_dataset('./train_Xc.csv')
# featurizing the test data
MAT_dataset_test = MAT_data_loader.create_dataset('./test_Xc.csv')

# Evaluating the model
metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
# setting up the loss storage
losses, val_losses = [], []

# initializing the model class
MAT_model = dc.models.torch_models.MATModel(batch_size=10)
for n_epoch in range(20):
    l2_loss_MAT = MAT_model.fit(MAT_dataset, nb_epoch=1)
    losses.append(l2_loss_MAT)
    val_losses.append(MAT_model.evaluate(MAT_dataset_test, [metric]))



# setting up the featurizer
dmpnn_featurizer = dc.feat.DMPNNFeaturizer()

# featurizing and preparing the dataset
dmpnn_loader = dc.data.CSVLoader(tasks=['value'], feature_field='smiles', featurizer=dmpnn_featurizer)
dmpnn_dataset = dmpnn_loader.create_dataset('./train_Xc.csv')
dmpnn_dataset_test = dmpnn_loader.create_dataset('./test_Xc.csv')
metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

losses_dmpnn, val_losses_dmpnn = [], []

# initializing the model class
dmpnn_model = dc.models.torch_models.DMPNNModel(batch_size=10)


for n in range(20):
    dmpnn_loss = dmpnn_model.fit(dmpnn_dataset, nb_epoch=1)
    losses_dmpnn.append(dmpnn_loss)
    dmpnn_val_loss = dmpnn_model.evaluate(dmpnn_dataset_test, [metric])
    val_losses_dmpnn.append(dmpnn_val_loss)


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

line1 = ax.plot(range(20), losses, label='Training Loss MAT', color='red', marker='o')
line2 = ax.plot(range(20), losses_gcn, label='Training Loss GCN', color='blue', marker='o')
line3 = ax.plot(range(20), losses_dmpnn, label='Training Loss D-MPNN', color='green', marker='o')

legend = ax.legend(loc='upper right', shadow=True)

custom_x_ticks = np.arange(0, 20, 1)
custom_x_label = [str(i+1) for i in custom_x_ticks]

ax.set_xticks(custom_x_ticks)
ax.set_xticklabels(custom_x_label)

plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Across Models')
plt.grid(True)
plt.show()