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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_recall_curve,
    auc,
)

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaseModel(pl.LightningModule):
    def __init__(self,is_pretrain=True):
        super(BaseModel, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = None
        self.decoder = None
        self.fc = None

        self.is_pretrain = is_pretrain

    def forward_pretrain(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        # x = x.view(x.size(0), -1)
        if self.is_pretrain:
            x_hat = self.forward_pretrain(x)
            loss = nn.functional.mse_loss(x_hat, x)

        else:
            pred = self.forward(x)
            loss = nn.BCEWithLogitsLoss()(pred, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        # x = x.view(x.size(0), -1)
        outputs = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)

        preds = torch.sigmoid(outputs)
        preds_binary = (preds > 0.5).float()

        accuracy = accuracy_score(labels.cpu().numpy(), preds_binary.cpu().numpy())
        recall = recall_score(
            labels.cpu().numpy(), preds_binary.cpu().numpy(), average="micro"
        )
        precision, recall, _ = precision_recall_curve(
            labels.cpu().numpy().ravel(), preds.cpu().detach().numpy().ravel()
        )
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("accuracy", accuracy)
        self.log("recall", recall.mean())
        self.log("precision", precision.mean())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class GRUModel(BaseModel):
    def __init__(self, input_size=1024, hidden_size=256, num_classes=12,is_pretrain=True):
        super().__init__(is_pretrain)

        self.input_size = input_size
        # 编码器部分：单层GRU
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        # 解码器部分：单层GRU
        self.decoder = nn.GRU(hidden_size, input_size, batch_first=True)
        # 输出层（线性变换）
        self.fc_out = nn.Linear(hidden_size, input_size)

        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, h = self.encoder(x)
        out = self.fc(out)
        return out

    def forward_pretrain(self, x):
        # 编码阶段
        out, h = self.encoder(x)
        # encoded = h.squeeze(0)  # 取出最后一个时间步的隐藏状态作为编码结果

        # # 解码阶段
        # decoded = self.fc_out(
        #     encoded.unsqueeze(0)
        # )  # 将编码结果重塑为(batch_size, 1, hidden_size)以适应GRU输入格式
        decoded, _ = self.decoder(out)
        return decoded

        # return decoded.reshape(-1, self.input_size)  # 返回重构后的图像向量

        # Logging to TensorBoard (if installed) by default


class TransformerModel(BaseModel):
    def __init__(
        self,
        input_size=1024,
        num_layers=2,
        num_heads=8,
        hidden_size=512,
        dropout=0.1,
        is_pretrain=True,
    ):
        super().__init__(is_pretrain)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

        self.fc = nn.Linear(input_size, 12)

class DNNSimpleModel(BaseModel):
    def __init__(self, input_size=1024, hidden_size=256, output_size=1024,is_pretrain=True):
        super().__init__(is_pretrain)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        self.fc = nn.Linear(output_size, 12)


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
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        # self.mask = torch.rand(self.x_train.shape[:2], dtype=torch.bool) < 0.4
        self.mask = (torch.rand(self.x_train.shape[:2]) < 0.4).bool()
        # self.seq_length = seq_length
        # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        mask = self.mask[idx]
        return x, mask
        # input_seq = self.x_train[idx][: self.seq_length]
        # target = self.x_train[idx][self.seq_length :]
        # input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        # target_tensor = torch.tensor(target, dtype=torch.float32)
        # return input_tensor.unsqueeze(0), target_tensor.unsqueeze(0)


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


class SmilesTrainDataset(Dataset):
    def __init__(self, data_path,is_train=True):
        self.data = pd.read_csv(data_path)
        feature_list = [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ]
        self.data.dropna(subset=feature_list,inplace=True)

        self.data["SMILES"] = self.data["SMILES"].apply(preprocess_smiles)

        X = np.stack(self.data["SMILES"].values)
        y = self.data[feature_list]   
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        #Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42)   
        
        # Convert SMILES to tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        if is_train:
            self.x_data = X_train_tensor
            self.y_data = y_train_tensor

        else:
            self.x_data = X_test_tensor
            self.y_data = y_test_tensor

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]