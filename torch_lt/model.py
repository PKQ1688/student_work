#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/31 23:51
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/31 23:51
# @File         : model.py
import torch

# import pytorch_lightning as pl
import lightning as pl
import torch.nn as nn


class GRUModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            embedding_dim, num_heads, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out


class DNNSimpleModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dnn = nn.Sequential(
            *[nn.Linear(embedding_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)],
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        x = self.embedding(x)
        out = self.dnn(x[:, -1, :])
        return out


class ModelDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data=None, test_data=None, batch_size=16):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage=None):
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
