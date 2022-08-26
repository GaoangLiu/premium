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

from premium.models.nn import BiLSTM, MultiClassifier, NNTouchStone

filepath = '/tmp/1000.csv'
filepath = '/tmp/imdb_sentiment.csv'
filepath = '/tmp/waimai_10k.csv'
filepath = '/tmp/online_shopping_10_cats.csv'
df = pd.read_csv(filepath)
df.dropna(inplace=True)
df['text'] = df.review
df['target'] = df.cat
if 0:
    from premium.experimental.myfasttext import benchmark
    benchmark(df)
clf = BiLSTM(max_feature=20000,
             max_length=200,
             vectorizer_split_strategy='character')

clf = MultiClassifier(max_feature=20000,
                      max_length=200,
                      vectorizer_split_strategy='character')

clf.benchmark(df, epochs=3, batch_size=64)
