#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codefast as cf

from utils.exceptions import FileFormatError


def read_conll_corpus(filename: str):
    """
    Read a corpus file with a format used in CoNLL.
    `data` demo:
    [
        ([['Friday', 'NNP'], ["'s", 'POS'], ['Market', 'NNP'], ['Activity', 'NN']], ['B-NP', 'B-NP', 'I-NP', 'I-NP'])
        ...
    ]
    """
    element_size = 0
    data, X, Y = [], [], []
    for s in cf.io.read(filename):
        words = [e for e in s.strip().split() if e]
        if not words:
            data.append((X, Y))
            X, Y = [], []
        else:
            if element_size == 0:
                element_size = len(words)
            elif element_size == len(words):
                X.append(words[:-1])
                Y.append(words[-1])
    if len(X) > 0:
        data.append((X, Y))

    return data


# fn = 'data/chunking_small/small_train.data'
# data = read_conll_corpus(fn)
# for d in data:
#     if len(d[0]) < 5:
#         print(d)
