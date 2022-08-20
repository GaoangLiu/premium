#!/usr/bin/env python
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import pandas as pd


def mda():
    # masked data angumentation
    from transformers import pipeline

    nlp = pipeline('fill-mask', model='bert-base-chinese')
    # xs = nlp('送货速度挺[MASK]的')

    xs = nlp('我我我的[MASK][MASK]就是做这个的')
    print(xs)


import fasttext


def benchmark(df: pd.DataFrame,
              dim: int = 200,
              pretrainedVectors: str = None,
              model_path: str = None,
              *args):
    cf.info('start training')
    train_args = {
        'input': '/tmp/tt.train',
        'dim': dim,
        'minn': 1,
        'maxn': 7,
        'thread': 12
    }

    if pretrainedVectors:
        check_pretrained_vector(pretrainedVectors)
        train_args['pretrainedVectors'] = pretrainedVectors
    model = fasttext.train_supervised(**train_args)
    model.save_model(model_path)

    # validate the model
    res = model.test("/tmp/tt.test")
    cf.info('validate result', res)
    return model


mda()
df = pd.read_csv('/tmp/train.csv')
df['text'] = df.text.apply(lambda x: ' '.join(list(x)))

# benchmark(df, dim=100, pretrainedVectors='/tmp/tencent100.txt')

try_bert = False
if try_bert:
    from premium.models.bert import bert_benchmark
    bert_benchmark(df, bert_name='bert-base-chinese', epochs=5)
