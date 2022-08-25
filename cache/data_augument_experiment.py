#!/usr/bin/env python
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from typing import Union

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


def benchmarklocal(df: pd.DataFrame,
                   dim: int = 200,
                   pretrainedVectors: str = None,
                   *args):
    cf.info('start training')
    train_args = {
        'input': 'tt.train',
        'dim': dim,
        'minn': 1,
        'maxn': 7,
        'thread': 12
    }

    if pretrainedVectors:
        train_args['pretrainedVectors'] = pretrainedVectors
    model = fasttext.train_supervised(**train_args)

    # validate the model
    res = model.test("tt.test")
    cf.info({'validate result': res})
    return model


import gensim
import jieba

from premium.corpus.stopwords import stopwords


def embedding_data_augument(df: pd.DataFrame,
                            pretrainedVectors: str) -> pd.DataFrame:
    ''' replace word with topn similar words in a sentence to generate more samples
    '''
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrainedVectors)
    cf.info('pretrained vector {} loaded'.format(pretrainedVectors))

    def enhence(text: str, target: Union[str, int], ratio: float = 0.2):
        # choose a word from sentence to replace with ratio
        text_list = []
        words = jieba.lcut(text)
        for i, w in enumerate(words):
            if w in stopwords().cn_stopwords: continue
            if w not in model: continue
            if random.random() > ratio: continue
            topn = model.most_similar(w)
            for nw, _ in topn:
                xs = words[:]
                xs[i] = nw
                text_list.append(''.join(xs))
        return text_list, [target] * len(text_list)

    texts, targets = [], []

    for t, l in zip(df.text.to_list(), df.target.to_list()):
        ts, ls = enhence(t, l)
        texts.extend(ts)
        targets.extend(ls)
    return pd.DataFrame(list(zip(texts, targets)), columns=['text', 'target'])


# mda()
fp = '/tmp/train.csv'
df = pd.read_csv(fp)
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, random_state=7)

# train = embedding_data_augument(train, '/tmp/tencent100.txt')


def format_to_fasttext_input(df: pd.DataFrame, save_path: str):
    xs = []
    for idx, row in df.iterrows():
        xs.append('__label__' + str(row['target']) + ' ' + row.text)
    cf.io.write('\n'.join(xs), save_path)


format_to_fasttext_input(train, 'tt.train')
format_to_fasttext_input(test, 'tt.test')
# print(df.head())
# df.to_csv('x.csv',index=False)
# df['text'] = df.text.apply(lambda x: ' '.join(list(x)))

from premium.experimental.myfasttext import benchmark

benchmarklocal(df, dim=100, pretrainedVectors='/tmp/tencent100.txt')

try_bert = False
if try_bert:
    from premium.models.bert import bert_benchmark
    bert_benchmark(df, bert_name='bert-base-chinese', epochs=5)
