#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/28 16:29
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/28 16:29
# @File         : handle_data.py
import os
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def save_train_data_list():
    file_name_list = os.listdir("data_v2")
    train_data_list = []
    val_data_list = []

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long
    )
    for file_name in file_name_list:
        # print(file_name)
        label = file_name.split("_")[0]
        label = torch.tensor([int(label)], dtype=torch.long)
        label = torch.nn.functional.one_hot(label, num_classes=5).float()

        # print(label)
        file_content = pd.read_csv(f"data_v2/{file_name}", header=None)
        file_np = np.array(file_content.values, dtype=np.float32)
        # print(file_np.shape)
        for i in tqdm(range(file_np.shape[0])):
            # for i in tqdm(range(1000)):
            inputs = torch.tensor(file_np[i])
            # print(inputs.shape)
            inputs = inputs.unsqueeze(1)
            data = Data(x=inputs, y=label, edge_index=edge_index)
            if random.random() < 0.7:
                train_data_list.append(data)
            else:
                val_data_list.append(data)
    # if is_train:
    #     self.data_list = train_data_list
    # else:
    #     self.data_list = val_data_list

    random.shuffle(train_data_list)
    random.shuffle(val_data_list)
    torch.save(train_data_list, "data_list_train.pt")
    torch.save(val_data_list, "data_list_val.pt")


if __name__ == '__main__':
    save_train_data_list()
