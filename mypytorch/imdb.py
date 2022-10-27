#!/usr/bin/env python
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb#scrollTo=1OXNyH3QzwT5
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
import numpy as np
import pandas as pd
import torch
import torchtext as tt
from rich import print
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_WORDS = 10000     # imdb’s vocab_size 即词汇表大小
MAX_LEN = 200     # max length
BATCH_SIZE = 256
EMB_SIZE = 128     # embedding size
HID_SIZE = 128     # lstm hidden size
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 借助Keras加载imdb数据集
df = pd.read_csv('/tmp/imdb_sentiment.csv').sample(1000)
x_train, x_test, y_train, y_test = train_test_split(df['review'],
                                                    df['sentiment'])

from premium.data.preprocess import tokenize
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<unk>')
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test=tokenizer.transform(x_test)

x_train = pad_sequences(x_train,
                        maxlen=MAX_LEN,
                        padding="post",
                        truncating="post")
x_test = pad_sequences(x_test,
                       maxlen=MAX_LEN,
                       padding="post",
                       truncating="post")
print(x_train.shape, x_test.shape)
