#!/usr/bin/env python
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb#scrollTo=1OXNyH3QzwT5
import os
import random
import json
import re
import sys
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import numpy as np
import pandas as pd
from rich import print
from typing import List, Union, Callable, Set, Dict, Tuple, Optional

import torch as T
import torchtext as tt
import numpy as np

from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, csv_file: str = '/tmp/imdb_sentiment.csv'):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tt.data.get_tokenizer('basic_english')
        self.vocab = tt.vocab.build_vocab_from_iterator(
            map(self.tokenizer, self.df['review']))
        self.vocab.set_default_index(self.vocab['the'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['review'][idx]
        label = self.df['sentiment'][idx]
        text = self.tokenizer(text)
        text = [self.vocab[token] for token in text]

        return text, label


tokenizer = tt.data.get_tokenizer('basic_english')
text = "are you okay, my friend? I'm fine."
print(tokenizer(text))

# training_data = CustomDataset()
# train_dataloader = DataLoader(training_data,
#                               batch_size=64,
#                               shuffle=True,
#                               collate_fn=lambda x: x)

# train_features, train_labels = next(iter(train_dataloader))
# print(train_features, train_labels)
