#!/usr/bin/env python
import random
from typing import Union

import codefast as cf
import gensim
import jieba
import pandas as pd

from premium.corpus.stopwords import stopwords


def embedding_data_augument(df: pd.DataFrame,
                            pretrainedVectors: str) -> pd.DataFrame:
    '''choose topn similar words to replace a word to generate new text
    '''                                
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrainedVectors)
    cf.info('pretrained vector {} loaded'.format(pretrainedVectors))

    def enhence(text: str, target: Union[str, int], ratio: float = 0.3):
        # choose a word from sentence to replace with ratio
        text_list = []
        words = jieba.lcut(text)
        for i, w in enumerate(words):
            if w in stopwords().cn_stopwords: continue
            if w not in model: continue
            if random.random() > ratio: continue
            topn = model.most_similar(w, topn=3)
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
        texts.append(t)
        targets.append(l)
    return pd.DataFrame(list(zip(texts, targets)), columns=['text', 'target'])


def aeda(df: pd.DataFrame) -> pd.DataFrame:
    # An easier data augumentation
    puncs = list('，‘’“”。！？：（）、《》')
    texts, targets = [], []
    for t, l in zip(df.text.to_list(), df.target.to_list()):
        ts = []
        for _ in range(5):
            tlist = list(t)
            idxs = random.sample(range(len(t)), 3)
            for idx in idxs:
                tlist.insert(idx, random.choice(puncs))
            ts.append(''.join(tlist))
        texts.extend(ts)
        targets.extend([l] * len(ts))
        texts.append(t)
        targets.append(l)
    return pd.DataFrame(list(zip(texts, targets)), columns=['text', 'target'])
