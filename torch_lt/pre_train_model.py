#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/31 17:46
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/31 17:46
# @File         : pre_train_model.py
# import pytorch_lightning as pl
import lightning as pl
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from model import GRUModel, TransformerModel, DNNSimpleModel, ModelDataModule


def preprocess_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)



class SmilesPreTrainDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data["smiles"] = self.data["smiles"].apply(preprocess_smiles)

        x_train = np.stack(self.data["smiles"].values)
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(x_train)
        # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float32)


# Experiment parameters
vocab_size = 1024
embedding_dim = 256
hidden_dim = 64
num_heads = 8
num_layers_gru = 2
num_layers_transformer = 1
num_layers_dnn = 3
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

# Data preparation (assuming you have train_data, val_data, and test_data ready)
train_data = SmilesPreTrainDataset("data/pre_train.csv")


data_module = ModelDataModule(train_data,batch_size)

# Model instantiation
gru_model = GRUModel(vocab_size, embedding_dim, hidden_dim, num_layers=num_layers_gru)
transformer_model = TransformerModel(vocab_size, embedding_dim, num_heads, num_layers=num_layers_transformer)
dnn_model = DNNSimpleModel(vocab_size, embedding_dim, hidden_dim, num_layers=num_layers_dnn)

# Trainer configuration
trainer = pl.Trainer(max_epochs=num_epochs, gpus=torch.cuda.device_count(), progress_bar_refresh_rate=20)

# Training and evaluation
trainer.fit(gru_model, datamodule=data_module)
trainer.fit(transformer_model, datamodule=data_module)
trainer.fit(dnn_model, datamodule=data_module)

# Compare models based on their validation or test metrics
# You can access the metrics using gru_results, transformer_results, and dnn_results
