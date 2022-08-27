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
filepath = 'localdata/tmp.csv'
filepath = '/tmp/twitter_disaster.csv'
df = pd.read_csv(filepath)
df.dropna(inplace=True)
# df['text'] = df.review
# df['target'] = df.sentiment
if 0:
    from premium.experimental.myfasttext import benchmark
    benchmark(df)

clf = MultiClassifier(max_feature=30000,
                      max_length=200,
                      vectorizer_split_strategy='character')

clf = BinClassifier(
    max_feature=20000,
    max_length=300,
    embedding_dim=100,
     # pretrained_vector_path='glove.twitter.27B.25d.txt',
    pretrained_vector_path='glove.6B.100d.txt',
    vectorizer_split_strategy='character')
clf.benchmark(df, epochs=10, batch_size=32)
