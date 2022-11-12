#!/usr/bin/env python
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb#scrollTo=1OXNyH3QzwT5
from typing import Callable

import codefast as cf
import pandas as pd
import torch
from rich import print
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch import nn

from premium.data.loader import imdb_sentiment
from premium.pytorch.tokenizer import VocabTokenizer
from premium.pytorch import BatchCollector
from premium.pytorch.trainer import Trainer


class Configs(object):
    MAX_WORDS = 10000
    MAX_LEN = 100
    BATCH_SIZE = 256
    EMB_SIZE = 128
    HID_SIZE = 128
    DROPOUT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(object):
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
        return {'text': tokens, 'label': label}


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
tokenizer.fit(df.text.tolist())
tokenizer.save('/tmp/myvocab.pt')
tokenizer.load('/tmp/myvocab.pt')
vocab_size = len(tokenizer)
Configs.MAX_WORDS = vocab_size + 2
print('vocab size ', len(tokenizer))
# print(tokenizer.['the'])
        

num_class = 2
device = Configs.DEVICE
emsize = 64
model = TextClassificationModel(vocab_size + 10, 1, emsize, 128, 1)
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
epoch = 3
train_dataloader = DataLoader(MyDataset(X, tokenizer), batch_size=Configs.BATCH_SIZE,
                              shuffle=True, collate_fn=BatchCollector(device=device))

"""
TODO
1. add validation
2. add test
"""

trner = Trainer(model, criterion, optimizer, scheduler, device)
trner.train(train_dataloader, 2)

