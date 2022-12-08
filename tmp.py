#!/usr/bin/env python
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from typing import List, Tuple

import codefast as cf
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich import print
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from premium.data.datasets import downloader
from premium.utils import cf_unless_tmp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MnistData(Dataset):

    def __init__(self, csv_file: str) -> None:
        if not cf.io.exists(csv_file):
            url = "https://pjreddie.com/media/files/{}".format(
                cf.io.basename(csv_file))
            cf.net.download(url, csv_file)
        #
        self.train = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        label = self.train.iloc[index, 0]
        target = torch.zeros(10)  # no difference to torch.zeros((10))
        target[label] = 1.0
        image = torch.FloatTensor(self.train.iloc[index, 1:].values) / 255.0
        return label, image, target


class Classifier(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 output_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 drop_prob: float = 0.5):
        """ Initialize the model by setting up the layers
        Args:
            vocab_size: the number of tokens in the vocabulary
            output_size: the desired size of the output
            embedding_dim: the size of the embedding layer
            hidden_dim: the size of the hidden layer outputs
            n_layers: the number of LSTM layers
            drop_prob: the dropout probability
        """
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=drop_prob,
                            batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # Linear and sigmoid layer
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size()

        # Embadding and LSTM output
        embedd = self.embedding(x)
        lstm_out, hidden = self.lstm(embedd, hidden)

        # stack up the lstm output
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layers
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        sig_out = self.sigmoid(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initialize Hidden STATE"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size,
                             self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size,
                             self.hidden_dim).zero_())

        return hidden