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

from premium.experimental.myfasttext import benchmark
from premium.models.nn import NNTouchStone

filepath = '/tmp/imdb_sentiment.csv'
df = pd.read_csv(filepath)
df = df.sample(frac=0.1)
targets = df.sentiment.values
# df['text'] = df.review
# df['target'] = targets
# benchmark(df)
clf = NNTouchStone(df.review, targets, max_length=200)
clf.pipeline()
