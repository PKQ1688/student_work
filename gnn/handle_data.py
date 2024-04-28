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


def save_train_data_list():
    file_name_list = os.listdir("data_v2")
    data_list = []
    for file_name in file_name_list:
        # print(file_name)
        label = file_name.split("_")[0]
        # print(label)
        file_content = pd.read_csv(f"data_v2/{file_name}", header=None)
        file_np = np.array(file_content.values)
        # print(file_np.shape)
        for i in range(file_np.shape[0]):
            inputs = torch.tensor(file_np[0])
            # print(inputs.shape)
            label = file_name.split()
            data = Data(x=inputs, y=label)
            data_list.append(data)
        # print(len(data_list))
        # break
    random.shuffle(data_list)
    data_list_train = data_list[:int(len(data_list) * 0.7)]
    data_list_val = data_list[int(len(data_list) * 0.7):]
    torch.save(data_list_train, "data_list_train.pt")
    torch.save(data_list_val, "data_list_val.pt")


if __name__ == '__main__':
    save_train_data_list()
