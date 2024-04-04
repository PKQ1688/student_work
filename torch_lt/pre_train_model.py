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
# vocab_size = 1024
# embedding_dim = 256
# hidden_dim = 64

# input_size = 1024
# hidden_size = 256
# output_size = 64
batch_size = 64
learning_rate = 1e-3
num_epochs = 10


# Model instantiation
gru_model = GRUModel()
transformer_model = TransformerModel()
dnn_model = DNNSimpleModel()

train_dataset = SmilesPreTrainDataset(data_path="torch_lt/merged_simples.csv")

# Trainer configuration
trainer_transformer = pl.Trainer(
    max_epochs=num_epochs, devices="auto", default_root_dir="torch_lt/transfomer/"
)
trainer_dnn = pl.Trainer(
    max_epochs=num_epochs, devices="auto", default_root_dir="torch_lt/dnn/"
)
trainer_gru = pl.Trainer(
    max_epochs=num_epochs, devices="auto", default_root_dir="torch_lt/gru"
)

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

# Training and evaluation
# trainer.fit(gru_model,data_loader)
trainer_transformer.fit(transformer_model, dataloader)
trainer_dnn.fit(dnn_model, dataloader)
trainer_gru.fit(gru_model, dataloader)
# trainer.fit(dnn_model, data_loader)

# Compare models based on their validation or test metrics
# You can access the metrics using gru_results, transformer_results, and dnn_results
