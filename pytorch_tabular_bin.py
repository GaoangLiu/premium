#!/usr/bin/env python3
import json
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from premium.utils import cf_unless_tmp


class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class BinaryClassification(nn.Module):

    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(12, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


class LitBinaryClassification(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = BinaryClassification()
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_true = y.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_true)
        acc = self.accuracy(y_hat, y_true)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        cf.info(
            f'Epoch {self.current_epoch}: | Loss: {avg_loss:.5f} | Acc: {avg_acc:.3f}'
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_true = y.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_true)
        acc = self.accuracy(y_hat, y_true)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        cf.info(
            f'Epoch {self.current_epoch}: | Val Loss: {avg_loss:.5f} | Val Acc: {avg_acc:.3f}'
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_true = y.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_true)
        acc = self.accuracy(y_hat, y_true)
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        cf.info(
            f'Epoch {self.current_epoch}: | Test Loss: {avg_loss:.5f} | Test Acc: {avg_acc:.3f}'
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return valid_loader

    def test_dataloader(self):
        return valid_loader


if __name__ == '__main__':
    filename = 'spine_dataset_300.csv'

    cf_unless_tmp(filename)
    df = pd.read_csv(f'/tmp/{filename}')
    print(df)
    X = df.iloc[:, 0:12]
    y = df.iloc[:, 12]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=69)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    EPOCHS = 30
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    train_data = TrainData(torch.FloatTensor(X_train),
                           torch.FloatTensor(y_train))
    valid_data = TrainData(torch.FloatTensor(X_test),
                           torch.FloatTensor(y_test.to_numpy()))
    test_data = TestData(torch.FloatTensor(X_test))

    train_loader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    model = LitBinaryClassification()
    trainer = pl.Trainer(max_epochs=EPOCHS, enable_progress_bar=False)
    # trainer.fit(model)
    # torch.save(model.state_dict(), 'trained_model.pth')

    model.load_state_dict(torch.load('trained_model.pth'))
    preds = trainer.predict(model, test_loader)
    print(preds)
