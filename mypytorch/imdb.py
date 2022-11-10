#!/usr/bin/env python
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb#scrollTo=1OXNyH3QzwT5

import math
from abc import ABC, abstractmethod
from functools import partial
import time
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import numpy as np
import pandas as pd
import torch
import torchtext as tt
from rich import print
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
from torch import nn

from premium.data.loader import imdb_sentiment
from premium.pytorch.tokenizer import VocabTokenizer
from premium.pytorch import BatchCollector
from tqdm import tqdm


class Configs(object):
    MAX_WORDS = 10000
    MAX_LEN = 100
    BATCH_SIZE = 256
    EMB_SIZE = 128
    HID_SIZE = 128
    DROPOUT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMDBDataset(object):
    def __init__(self, df: pd.DataFrame, tokenizer: Callable):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = Configs.MAX_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = row['label']
        tokens = self.tokenizer(text)[0]
        tokens.extend([0] * (self.max_len - len(tokens)))
        tokens = tokens[:self.max_len]
        return {'text': torch.tensor(tokens), 'label': label}


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


df = imdb_sentiment().train
df.rename(columns={'review': 'text', 'sentiment': 'label'}, inplace=True)
X, V = train_test_split(df, test_size=0.2, random_state=42)
V, T = train_test_split(V, test_size=0.5, random_state=42)
print(T)

tokenizer = VocabTokenizer()
tokenizer.load('/tmp/myvocab.pt')
vocab_size = len(tokenizer)
Configs.MAX_WORDS = vocab_size + 2
print('vocab size ', len(tokenizer))
resp = tokenizer('the tortuous emotional impact is degrading')
print(resp)
# print(tokenizer.['the'])

num_class = 2
device = Configs.DEVICE
emsize = 64
model = TextClassificationModel(vocab_size + 10, emsize, num_class)
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
epoch = 3
train_dataloader = DataLoader(IMDBDataset(X, tokenizer), batch_size=Configs.BATCH_SIZE,
                              shuffle=True, collate_fn=BatchCollector(device=device))

"""
TODO
1. wandb
2. tensor pad 
"""
class TrainerInterface(ABC):
    @abstractmethod
    def evaluate(self, valid_dataloader):
        pass

    @abstractmethod
    def train(self, train_dataloader, epoch):
        pass


class Metrics(object):
    def __init__(self, epoch: int):
        self.loss = 0
        self.n_correct = 0
        self.n_total = 0
        self.cur_loss = 0
        self.cur_acc = 0
        self.epoch = epoch

    def update(self, loss, n_correct, n_total):
        self.loss += loss
        self.n_correct += n_correct
        self.n_total += n_total
        self.calc()

    def calc(self):
        self.cur_loss = self.loss / self.n_total
        self.cur_acc = self.n_correct / self.n_total
        return self.cur_loss, self.cur_acc

    def __str__(self):
        return f'| epoch {self.epoch:3d} | loss {self.cur_loss:5.2f} | acc {self.cur_acc:5.2f}'


class Trainer(TrainerInterface):
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_dataloader, epoch):
        self.model.to(self.device)
        self.model.train()

        for e in range(epoch):
            metric = Metrics(e)
            for batch in tqdm(train_dataloader):
                text, offsets, label = batch['text'], batch['offset'], batch['label']
                self.optimizer.zero_grad()
                output = self.model(text, offsets)
                loss = self.criterion(output, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                metric.update(loss.item(), (output.argmax(1) == label).sum().item(), label.size(0))

            cf.info(metric)

    def evaluate(self, valid_dataloader):
        pass


trner = Trainer(model, criterion, optimizer, scheduler, device)
trner.train(train_dataloader, 2)


# def train(dataloader):
#     model.to(device)
#     model.train()
#     total_acc, total_count = 0, 0
#     log_interval = 50
#     start_time = time.time()
#     for _epoch in range(epoch):

#         for idx, batch in enumerate(dataloader):
#             text = batch['text']
#             label = batch['label']
#             offsets = batch['offset']
#             optimizer.zero_grad()

#             predicted_label = model(text, offsets)
#             loss = criterion(predicted_label, label)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#             optimizer.step()
#             total_acc += (predicted_label.argmax(1) == label).sum().item()
#             total_count += label.size(0)
#             if idx % log_interval == 0 and idx > 0:
#                 elapsed = time.time() - start_time
#                 print('| epoch {:3d} | {:5d}/{:5d} batches '
#                       '| accuracy {:8.3f}'.format(_epoch, idx, len(dataloader),
#                                                   total_acc/total_count))
#                 total_acc, total_count = 0, 0
#                 start_time = time.time()


# train(train_dataloader)
