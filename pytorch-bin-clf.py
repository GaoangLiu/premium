#!/usr/bin/env python
from torch.utils.data import TensorDataset, DataLoader
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
from torch.utils.data import Dataset, DataLoader


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
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):    
        """
        Initialize the model by setting up the layers
        """
        super().__init__()
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        
        #Embedding and LSTM layers
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        #dropout layer
        self.dropout=nn.Dropout(0.3)
        
        #Linear and sigmoid layer
        self.fc1=nn.Linear(hidden_dim, 64)
        self.fc2=nn.Linear(64, 16)
        self.fc3=nn.Linear(16,output_size)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size=x.size()
        
        #Embadding and LSTM output
        embedd=self.embedding(x)
        lstm_out, hidden=self.lstm(embedd, hidden)
        
        #stack up the lstm output
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_dim)
        
        #dropout and fully connected layers
        out=self.dropout(lstm_out)
        out=self.fc1(out)
        out=self.dropout(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.fc3(out)
        sig_out=self.sigmoid(out)
        
        sig_out=sig_out.view(batch_size, -1)
        sig_out=sig_out[:, -1]
        
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize Hidden STATE"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        
        return hidden


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

    def transform(self, texts: List[str]) -> np.ndarray:
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
    train_data = TensorDataset(torch.from_numpy(vecs), torch.from_numpy(targets))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    vocab_size = len(tokenizer.vocab)
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = Classifier(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)
    # batch_size = 50
    # model = Classifier(len(tokenizer.vocab), 32, 100, 1)
    # model.train(vecs, targets)
    
    lr=0.001
    batch_size=32

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip=5 # gradient clipping

    # move model to GPU, if available
    # net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            # inputs=inputs.cuda()
            # labels=labels.cuda()
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

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
                # net.eval()
                # for inputs, labels in valid_loader:

                #     # Creating new variables for the hidden state, otherwise
                #     # we'd backprop through the entire training history
                #     val_h = tuple([each.data for each in val_h])

                #     inputs, labels = inputs.cuda(), labels.cuda()  
                #     output, val_h = net(inputs, val_h)
                #     val_loss = criterion(output.squeeze(), labels.float())

                #     val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()))

        
