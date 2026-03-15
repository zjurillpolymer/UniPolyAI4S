import os
import logging
import warnings

import numpy as np
import torch

# 屏蔽干扰信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


'''mol==molecule'''


# all imports
from deepchem.molnet.load_function.zinc15_datasets import load_zinc15
from deepchem.feat.molecule_featurizers.raw_featurizer import RawFeaturizer
from deepchem.feat.molecule_featurizers.smiles_to_image import SmilesToImage  # this featurizer converts smiles string to image
from deepchem.data.datasets import Dataset
from rdkit import Chem
import matplotlib.pyplot as plt
import networkx as nx

raw_feat=RawFeaturizer()
zinc15_raw_data=load_zinc15(featurizer=raw_feat)




'''从数据集中提取信息'''
def get_training_data(dataset:Dataset,verbose=1):

    tasks,datasets,transformer=dataset
    print("tasks",tasks) if verbose else None
    training_data,valid_data,testing_data=datasets
    # if verbose:
    #     print("training_data",training_data)
    #     print("valid_data",valid_data)
    #     print("testing_data",testing_data)
    #     print("transformer",transformer)
    return training_data


def data_verbose(dataset:Dataset,verbose=1):
    traning_data=get_training_data(dataset,verbose)
    one_mol=None
    for (xi,yi,wi,idi) in traning_data.itersamples():
        one_mol=xi=xi
        if verbose:
            print("Molecule Object >>", xi)
            print("Task Target Value>>", yi)  # the target label for the dataset
            print("Weight >>", wi)  # weight associated
            print("ZINC ID>>", idi)  # zinc id for the molecule
        return one_mol
# print(data_verbose(zinc15_raw_data))
# # traning_data = get_training_data(zinc15_raw_data)
# # print(type(traning_data))    output:<class 'deepchem.data.datasets.DiskDataset'>


'''Monomer Molecules Representation'''
first_molecule=data_verbose(zinc15_raw_data)
print(first_molecule.GetNumAtoms(),end=' ')
print(first_molecule.GetNumBonds(),end=' ')
first_molecule.show()
