#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/6 16:03
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/6 16:03
# @File         : model.py
import os
import random
import pdb

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.transforms as T
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torchmetrics import (
    Accuracy,
    Recall,
    Precision,
    F1Score,
)
from multiprocessing import Pool

# from torch_geometric.data import Dataset, Batch

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, "Cora", transform=T.NormalizeFeatures())
batch_size = 1

def custom_collate(batch):
    batched_data = Batch.from_data_list(batch)
    return batched_data

def process_file(file_name,data_list):
    label = int(file_name.split("_")[0])
    label = torch.tensor([label], dtype=torch.long)
    label = torch.nn.functional.one_hot(label, num_classes=5).float()
    
    file_content = pd.read_csv(os.path.join("data_v2", file_name), header=None)
    # file_np = file_content.values.astype(np.float32)
    file_np = np.array(file_content.values, dtype=np.float32)
    
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
    
    # data_list = []
    # for i in range(file_np.shape[0]):
    # file_tensors = torch.tensor(file_np).unsqueeze(1)  # 处理一次性转换
    # for i in tqdm(range(30000)):
    # for i in tqdm(range(file_np.shape[0])):
    #     inputs = torch.tensor(file_np[i]).unsqueeze(1)
    for i in tqdm(range(0, file_np.shape[0], 1024)):
        # 使用切片获取当前批次的数据
        current_batch = file_np[i:i+1024]
    
        # 将 numpy 数组转换为 torch Tensor
        inputs = torch.tensor(current_batch)
        data = Data(x=inputs, y=label, edge_index=edge_index)
        # if random.random() < 0.7 if is_train else True:  # Always true for validation to include all data
        data_list.append(data)
    
    return data_list


class CustomDataset(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        # if is_train:
        #     self.data_list = torch.load("data_list_train.pt")
        # else:
        #     self.data_list = torch.load("data_list_val.pt")

        data_list = []
        file_name_list = os.listdir("data_v2")

        for file_name in file_name_list:
            print(file_name)
            process_file(file_name,data_list)  # Warm up the function

        random.shuffle(data_list)
        # Preparing training data
        # data_list = parallel_process_files(file_name_list)

        # Preparing validation data (assuming we don't need to reprocess files for validation, just split differently)
        # If validation requires a separate processing or different files, adjust accordingly.
        if is_train:
            self.data_list = data_list[:int(len(data_list) * 0.7)]  # Assuming we're in a class context, replace with appropriate variable if not
        else:
            self.data_list = data_list[int(len(data_list) * 0.7):]
        # file_name_list = os.listdir("data_v2")
        # train_data_list = []
        # val_data_list = []

        # edge_index = torch.tensor(
        #     [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long
        # )
        # for file_name in file_name_list:
        #     # print(file_name)
        #     label = file_name.split("_")[0]
        #     label = torch.tensor([int(label)], dtype=torch.long)
        #     label = torch.nn.functional.one_hot(label, num_classes=5).float()

        #     # print(label)
        #     file_content = pd.read_csv(f"data_v2/{file_name}", header=None)
        #     file_np = np.array(file_content.values, dtype=np.float32)
        #     # print(file_np.shape)
        #     for i in tqdm(range(file_np.shape[0])):
        #     # for i in tqdm(range(1000)):
        #         inputs = torch.tensor(file_np[i])
        #         # print(inputs.shape)
        #         inputs = inputs.unsqueeze(1)
        #         data = Data(x=inputs, y=label, edge_index=edge_index)
        #         if random.random() < 0.7:
        #             train_data_list.append(data)
        #         else:
        #             val_data_list.append(data)
        # if is_train:
        #     self.data_list = train_data_list
        # else:
        #     self.data_list = val_data_list

        # random.shuffle(self.data_list)

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


# class CustomGNNModel(nn.Module):
#     def __init__(self, in_channels=8, out_channels=5):
#         super().__init__()
#         self.conv1 = torch_geometric.nn.GCNConv(in_channels, 64)  # 第一层 GNN 层
#         self.conv2 = torch_geometric.nn.GCNConv(64, 256)  # 第二层 GNN 层
#         self.fc_out = nn.Linear(256, out_channels)  # 输出层，将 512 维特征映射到 5 维度矩阵
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = self.fc_out(x)
#         # 将输出特征重塑为 (batch_size, 24, 24) 形状
#         # x = x.view(-1, 24, 24)
#         # x = x.squeeze()
#         return x


class CustomGNNLightning(pl.LightningModule):
    def __init__(self, in_channels=1024, out_channels=5, hidden_channels=64):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(num_classes=5, task="multiclass",average="macro")
        self.precision = Precision(num_classes=5, task="multiclass",average="macro")
        self.recall = Recall(num_classes=5, task="multiclass",average="macro")
        self.f1 = F1Score(num_classes=5, task="multiclass",average="macro")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings

        # import pdb
        # pdb.set_trace()
        x = x.permute(1, 0)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # pdb.set_trace()
        # 2. Readout layer
        # x = torch_geometric.nn.global_mean_pool(
        #     x, batch
        # )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # x = F.softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)

        y_hat = F.softmax(y_hat, dim=1)
        predictions = torch.argmax(y_hat, dim=1)
        target = torch.argmax(batch.y, dim=1)
        # predictions = torch.nn.functional.one_hot(predictions, num_classes=5).float()

        # import pdb
        # pdb.set_trace()

        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)

        self.accuracy(predictions, target)
        self.precision(predictions, target)
        self.recall(predictions, target)
        self.f1(predictions, target)

        # accuracy = self.accuracy(predictions, target)
        # precision = self.precision(predictions, target)
        # recall = self.recall(predictions, target)
        # f1 = self.f1(predictions, target)
        
        # self.log("val_accuracy", accuracy, prog_bar=True, logger=True, batch_size=2048)
        # self.log("val_precision", precision, prog_bar=True, logger=True, batch_size=2048)
        # self.log("val_recall", recall, prog_bar=True, logger=True, batch_size=2048)
        # self.log("val_f1", f1, prog_bar=True, logger=True, batch_size=2048)

    def on_validation_epoch_end(self):
        # 计算整个验证集上的评价指标
        accuracy = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()

        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("val_precision", precision, prog_bar=True, logger=True)
        self.log("val_recall", recall, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)

        # 重置指标以备下一轮验证
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

        return accuracy, precision, recall, f1

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.005)
        return optimizer


if __name__ == "__main__":
    train_dataset = CustomDataset(is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate,
    )
    val_dataset = CustomDataset(is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate,
    )

    model = CustomGNNLightning()
    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="gcn")
    trainer = pl.Trainer(max_epochs=10, logger=True)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
