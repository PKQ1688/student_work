#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/7 10:59
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/7 10:59
# @File         : predict.py
import torch
# from torch_geometric.data import Data

from gnn.load_dataset import load_edge_index
from model import CustomGNNLightning,CustomDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomGNNLightning.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=4-step=470.ckpt",
                                                map_location=device)

model.to(device)
model.eval()

dataset = CustomDataset(data_list_path="data_list.pt")
# print(dataset[0])

# input_s = torch.rand(32, 7, 7).to(device)
output_s = torch.rand(24, 24)
edge_index = load_edge_index('connection_namelist.txt').to(device)

with torch.no_grad():
    # input_data = Data(x=input_s, y=output_s, edge_index=edge_index)
    input_data = dataset[0]
    output = model(input_data)
    # print(output.size())
    # print(output)
    print("模型预测结果：")
    print(output)
    print("真实结果：")
    print(input_data.y)
