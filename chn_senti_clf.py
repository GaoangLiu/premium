#!/usr/bin/env python
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
# https://huggingface.co/datasets/seamew/ChnSentiCorp
from datasets import load_dataset
from rich import print

from premium.data.datasets import downloader
from premium.experimental.eda import back_translation as BT
# from premium.experimental.myfasttext import baseline as ftbase
# from premium.models.bert import baseline as bb
from premium.models.nn import BinClassifier


def load(fpath: str) -> pd.DataFrame:
    return pd.read_csv(fpath, sep='\t', header=0)


def baseline():
    downloader.chn_senti()
    df = pd.read_csv('/tmp/chn_senti_corp.csv')
    df['target'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
    df['text'] = df.text.apply(lambda x: ' '.join(jieba.lcut(x)))
    client = BinClassifier()
    client.baseline(df, epochs=5)
    # df = pd.concat([train, dev], axis=0)
    # df['target'] = df['label']
    print(df.head())
    # bb(df, bert_name='bert-base-chinese', epochs=3)


def back_trans():
    # downloader.chn_senti()
    df = pd.read_csv('/tmp/chn_senti_corp.csv')
    tmp = df.sample(10)
    tmp['text'] = tmp.text.apply(BT)
    print(tmp)


if __name__ == '__main__':
    back_trans()
