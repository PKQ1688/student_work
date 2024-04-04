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
import torch.utils


from model import GRUModel, TransformerModel, DNNSimpleModel, SmilesPreTrainDataset

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

# Model instantiation
gru_model = GRUModel(vocab_size, embedding_dim, hidden_dim, num_layers=num_layers_gru)
# transformer_model = TransformerModel(vocab_size, embedding_dim, num_heads, num_layers=num_layers_transformer)
# dnn_model = DNNSimpleModel(vocab_size, embedding_dim, hidden_dim, num_layers=num_layers_dnn)

train_dataset = SmilesPreTrainDataset(data_path="merged_simples.csv")

# Trainer configuration
trainer = pl.Trainer(max_epochs=num_epochs, devices="auto")

# Training and evaluation
trainer.fit(gru_model,torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
# trainer.fit(transformer_model, datamodule=data_module)
# trainer.fit(dnn_model, datamodule=data_module)

# Compare models based on their validation or test metrics
# You can access the metrics using gru_results, transformer_results, and dnn_results
