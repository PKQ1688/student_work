#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/31 23:51
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/31 23:51
# @File         : model.py
import torch
import numpy as np
import pandas as pd

# import pytorch_lightning as pl
import lightning as pl
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class BaseModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(BaseModel, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.feature = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # embeds = self.embedding(x)
        output, _ = self.feature(x)
        # output = self.fc(output[:, -1])
        output = self.fc(
            output[:, -1, :]
        )  # Using only the output of the last time step
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        # loss = nn.CrossEntropyLoss()(y_pred, y)
        loss = nn.NLLLoss(y_pred.view(-1, 256), y.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class GRUModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__(vocab_size, embedding_dim, hidden_dim, num_layers=1)
        self.feature = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)


class TransformerModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers=1):
        super().__init__(vocab_size, embedding_dim, num_heads, num_layers=1)
        self.feature = nn.Transformer(
            embedding_dim, num_heads, num_layers, dropout=0.1, batch_first=True
        )


class DNNSimpleModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__(vocab_size, embedding_dim, hidden_dim, num_layers=2)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)


def preprocess_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


class SmilesPreTrainDataset(Dataset):
    def __init__(self, data_path, seq_length=256):
        self.data = pd.read_csv(data_path)
        self.data["smiles"] = self.data["smiles"].apply(preprocess_smiles)

        x_train = np.stack(self.data["smiles"].values)
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(x_train)
        self.seq_length = seq_length
        # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq = self.x_train[idx][: self.seq_length]
        target = self.x_train[idx][self.seq_length :]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return input_tensor.unsqueeze(0), target_tensor.unsqueeze(0)


class ModelDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data=None, test_data=None, batch_size=16):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SmilesPreTrainDataset(self.train_data)
        self.val_dataset = SmilesPreTrainDataset(self.val_data)
        self.test_dataset = SmilesPreTrainDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
