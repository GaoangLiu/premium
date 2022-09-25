#!/usr/bin/env python
import json
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import jieba
import joblib
import numpy as np
import pandas as pd
from rich import print


def to_train(words: List) -> List[Tuple]:
    res = []
    for w in words:
        if len(w) == 1:
            res.append((w, 'S'))
        else:
            res.append((w[0], 'B'))
            for c in w[1:-1]:
                res.append((c, 'M'))
            res.append((w[-1], 'E'))
    return res


texts = []
for ln in cf.io.read('/tmp/web'):
    js = json.loads(ln)
    texts.append(js['title'])
    texts.append(js['topic'])
    texts.extend(js['content'].split('。'))

# lst = list(cf.io.walk('/data/talks/yimutian'))
# random.shuffle(lst)
lst = texts

all_tokens = []

for f in lst[:100000]:
    f = f.replace(' ', '，')
    xs = to_train(jieba.lcut(f))
    all_tokens.append(xs)

train_s, test_s = '', ''
for i, lst in enumerate(all_tokens):
    s = '\n'.join([' '.join(e) for e in lst])
    if i > len(all_tokens) * 0.99:
        test_s += s + '\n\n'
    else:
        train_s += s + '\n\n'

cf.io.write(train_s, '/tmp/train_s')
cf.io.write(test_s, '/tmp/test_s')
