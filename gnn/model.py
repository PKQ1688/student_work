#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/6 16:03
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/6 16:03
# @File         : model.py

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset, Batch


# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, "Cora", transform=T.NormalizeFeatures())


def custom_collate(batch):
    batched_data = Batch.from_data_list(batch)
    return batched_data


class CustomDataset(Dataset):
    def __init__(self, data_list_path):
        super().__init__()
        self.data_list = torch.load(data_list_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = T.NormalizeFeatures()(data)
        return data
        # # 假设 `data` 是一个包含节点特征、边信息等的字典
        # x = data.x  # 节点特征，形状应为 (num_nodes, 32, 7, 7)
        # edge_index = data.edge_index  # 边的连接关系，形状应为 (2, num_edges)
        # y = data.y  # 输出标签，形状应为 (24, 24)
        # #
        # # # 将节点特征展平为 (num_nodes, 32 * 49) 形状，以便输入 GNN
        # # x_flattened = x.view(x.size(0), -1)
        # #
        # out = {
        #     'x': x,
        #     'edge_index': edge_index,
        #     'y': y
        # }
        #
        # return out


class CustomGNNModel(nn.Module):
    def __init__(self, in_channels=224, out_channels=576):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 64)  # 第一层 GNN 层
        self.conv2 = torch_geometric.nn.GCNConv(64, 256)  # 第二层 GNN 层
        self.fc_out = nn.Linear(7 * 256, out_channels)  # 输出层，将 512 维特征映射到 24*24 输出矩阵

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = x.view(-1, x.size(-1))
        x = x.transpose(0, 1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        # 将输出特征重塑为 (batch_size, 24, 24) 形状
        x = x.view(-1, 24, 24)
        x = x.squeeze()
        return x


class CustomGNNLightning(pl.LightningModule):
    def __init__(self, in_channels=224, out_channels=576):
        super().__init__()
        self.model = CustomGNNModel(in_channels=in_channels, out_channels=out_channels)
        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)  # 假设 `batch.y` 存储真实标签（7x7 矩阵）
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer


# 假设 `raw_data` 是包含所有样本数据的列表
dataset = CustomDataset(data_list_path="data_list.pt")
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate)

model = CustomGNNLightning()
trainer = pl.Trainer(max_epochs=5, logger=True)

trainer.fit(model, loader)
