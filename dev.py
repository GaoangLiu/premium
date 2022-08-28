#!/usr/bin/env python
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
from rich import print

from premium.models.nn import BinClassifier, MultiClassifier, NNTouchStone

filepath = '/tmp/1000.csv'
filepath = '/tmp/waimai_10k.csv'
filepath = '/tmp/online_shopping_10_cats.csv'
filepath = 'localdata/hemabot.csv'
# filepath = '/tmp/twitter_disaster.csv'
df = pd.read_csv(filepath)
df.dropna(inplace=True)
# df['text'] = df.review
# df['target'] = df.sentiment
if 0:
    from premium.experimental.myfasttext import benchmark
    benchmark(df)

clf = BinClassifier(
    max_feature=20000,
    max_length=300,
    embedding_dim=100,
     # pretrained_vector_path='glove.twitter.27B.25d.txt',
    pretrained_vector_path='glove.6B.100d.txt',
    vectorizer_split_strategy='character')
clf = MultiClassifier(max_feature=30000, max_length=200, embedding_dim=100)

model, _ = clf.benchmark(df, epochs=1, batch_size=32)


def load_test_data():
    lines = cf.io.read('localdata/test_hema.txt')
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    return lines


test_data = load_test_data()
ypreds = clf.predict(test_data)
ypreds = np.argmax(ypreds, axis=1)
print(ypreds)

ypreds = [clf.idx2target[i] for i in ypreds]
print(ypreds)
