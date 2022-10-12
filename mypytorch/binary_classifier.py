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
        target = torch.zeros(10)     # no difference to torch.zeros((10))
        target[label] = 1.0
        image = torch.FloatTensor(self.train.iloc[index, 1:].values) / 255.0
        return label, image, target


class Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.02),
        )
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.count = 0
        self.progress = []

    def forward(self, x):
        return self.model(x)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1
        self.progress.append(loss.item())
        if self.count % 10000 == 0:
            print(f"count {self.count}, avg loss {np.mean(self.progress)}")
            self.progress = []

    def predict(self, x):
        return self.forward(x).argmax(dim=1)


class Tokenizer(object):

    def __init__(self,
                 text_path: str,
                 max_words: int = 10000,
                 max_length: int = 100) -> None:
        """ 
        Input: 
            max_words: max words in vocab 
            max_length: max words in a sentence to keep 
        """
        self.text_path = text_path
        self.max_words = max_words
        self.max_length = max_length
        self.vocab = defaultdict(int)

    def load_data(self):
        df = pd.read_csv(self.text_path)
        X_train, X_test, y_train, y_test = train_test_split(
            df.text.values, df.target.values)
        return X_train, X_test, y_train, y_test

    def build_vocab(self, texts: List[str]) -> None:
        cf.info('start building vocab')
        for ln in texts:
            for token in ln.split(' '):
                self.vocab[token] += 1
        cf.info('build vocab completed')

    def transform(self, texts: List[str]) -> np.array:
        """ Transform list of text into list of vectors
        """
        if not self.vocab:
            self.build_vocab(texts)
        vectors = [[self.vocab[token]
                    for token in text.split(' ')][:self.max_length]
                   for text in texts]
        padded_vectors = np.array([
            vector + [0] * (self.max_length - len(vector)) for vector in vectors
        ])
        return padded_vectors


if __name__ == '__main__':
    filepath = '/tmp/imdb_sentiment.csv'
    df = pd.read_csv(filepath)
    df = df.sample(frac=0.1)
    targets = df.sentiment.values
    tokenizer = Tokenizer(filepath)
    vecs = tokenizer.transform(df.review)
    train_data = TensorDataset(
        torch.from_numpy(vecs), torch.from_numpy(targets))
    # dataloaders
    batch_size = 50

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print(vecs[:10])
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print('Sample input: \n', sample_y)
    
    clf = Classifier(100, 100, 1)
    for epoch in range(3):
        for inputs, targets in train_loader:
            clf.train(inputs, targets)
            
    