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
from premium.pytorch.data import TextDataset, TextLoader, train_test_val_split
from premium.pytorch.tokenizer import VocabVectorizer
from premium.utils import cf_unless_tmp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_dataloader(vectors: np.ndarray, targets: np.array):
    data = TensorDataset(torch.from_numpy(vectors), torch.from_numpy(targets))
    return DataLoader(data, shuffle=True, batch_size=32, drop_last=True)


if __name__ == '__main__':
    cf_unless_tmp('imdb_sentiment.csv')
    filepath = '/tmp/imdb_sentiment.csv'
    df = pd.read_csv(filepath)
    df = df.sample(frac=0.1)
    train, val, test = train_test_val_split(df)
    vectorizer = VocabVectorizer(max_length=100)
    train_vectors = vectorizer.fit_transform(train.review)
    val_vectors = vectorizer.transform(val.review)
    test_vectors = vectorizer.transform(test.review)

    train_loader = build_dataloader(train_vectors, train.sentiment.values)
    val_loader = build_dataloader(val_vectors, val.sentiment.values)
    test_loader = build_dataloader(test_vectors, test.sentiment.values)

    vocab_size = vectorizer.size()
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = Classifier(vocab_size, output_size, embedding_dim, hidden_dim,
                     n_layers)
    print(net)

    lr = 0.001
    batch_size = 32

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epochs = 10  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    net.to(device)
    net.train()
    # train for some number of epochs
    for e in range(epochs):
        h = net.init_hidden(batch_size)

        for inputs, labels in train_loader:
            counter += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            acc = np.mean((output > 0.5).cpu().numpy() == labels.cpu().numpy())

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                val_accuracy = []
                net.eval()
                for inputs, labels in val_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, labels = inputs.cuda(), labels.cuda()
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                net.train()
                msg = {
                    "epoch": "{}/{}".format(e + 1, epochs),
                    "step": counter,
                    "loss": round(loss.item(), 4),
                    "acc": acc,
                }
                cf.info(msg)
