#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/30 12:07
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/30 12:07
# @File         : load_dataset.py
import torch
from torch_geometric.data import Data


def load_x_y_feature(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    x_feature = []
    for line in lines:
        x_feature.append([float(i) for i in line.strip().split()])
    x_origin_feature = torch.tensor(x_feature, dtype=torch.float)
    # return x_origin_feature
    return x_origin_feature[0:7, 0:7]


def load_edge_index(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    _edge_index = []
    for line in lines:
        _edge_index.append([int(float(i)) for i in line.strip().split()])
    _edge_index = torch.tensor(_edge_index, dtype=torch.long)
    return _edge_index


def load_train_data():
    x_0 = load_x_y_feature('whole_datasets/x_0.txt')
    x_1 = load_x_y_feature('whole_datasets/x_1.txt')
    x_2 = load_x_y_feature('whole_datasets/x_2.txt')
    x_3 = load_x_y_feature('whole_datasets/x_3.txt')
    x_4 = load_x_y_feature('whole_datasets/x_4.txt')
    x_5 = load_x_y_feature('whole_datasets/x_5.txt')
    x_6 = load_x_y_feature('whole_datasets/x_6.txt')
    x_7 = load_x_y_feature('whole_datasets/x_7.txt')
    x_8 = load_x_y_feature('whole_datasets/x_8.txt')
    x_9 = load_x_y_feature('whole_datasets/x_9.txt')
    x_10 = load_x_y_feature('whole_datasets/x_10.txt')
    x_11 = load_x_y_feature('whole_datasets/x_11.txt')
    x_12 = load_x_y_feature('whole_datasets/x_12.txt')
    x_13 = load_x_y_feature('whole_datasets/x_13.txt')
    x_14 = load_x_y_feature('whole_datasets/x_14.txt')
    x_15 = load_x_y_feature('whole_datasets/x_15.txt')

    y_0 = load_x_y_feature('whole_datasets/y_0.txt')
    y_1 = load_x_y_feature('whole_datasets/y_1.txt')
    y_2 = load_x_y_feature('whole_datasets/y_2.txt')
    y_3 = load_x_y_feature('whole_datasets/y_3.txt')
    y_4 = load_x_y_feature('whole_datasets/y_4.txt')
    y_5 = load_x_y_feature('whole_datasets/y_5.txt')
    y_6 = load_x_y_feature('whole_datasets/y_6.txt')
    y_7 = load_x_y_feature('whole_datasets/y_7.txt')
    y_8 = load_x_y_feature('whole_datasets/y_8.txt')
    y_9 = load_x_y_feature('whole_datasets/y_9.txt')
    y_10 = load_x_y_feature('whole_datasets/y_10.txt')
    y_11 = load_x_y_feature('whole_datasets/y_11.txt')
    y_12 = load_x_y_feature('whole_datasets/y_12.txt')
    y_13 = load_x_y_feature('whole_datasets/y_13.txt')
    y_14 = load_x_y_feature('whole_datasets/y_14.txt')
    y_15 = load_x_y_feature('whole_datasets/y_15.txt')

    x = torch.stack([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15], dim=0)
    y = torch.stack([y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15], dim=0)

    edge_index = load_edge_index('connection_figure.txt')
    data = Data(x=x, y=y, edge_index=edge_index)

    return data


if __name__ == '__main__':
    # x_feature_0 = load_x_y_feature('whole_datasets/x_0.txt')
    # print(x_feature_0)
    # print(x_feature_0.shape)

    # edge_index_ = load_edge_index('connection_namelist.txt')
    # print(edge_index_)
    # print(edge_index_.shape)

    load_train_data()
