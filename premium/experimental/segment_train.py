#!/usr/bin/env python
""" Train a Chinese sentence segment model.
"""
import codefast as cf
from rich import print
from typing import List, Union, Callable, Set, Dict, Tuple, Optional

from premium.crf import CRF


def reformat(X: List[List[str]]) -> List[List[Tuple]]:
    """ Convert a list of list of words to a list of list of (word, label) tuples.
    Args:
        X: train corpus, a list of segmented sentences. E.g., [['我', '爱', '自由'], ['渴望', '民主']]
    Returns:
        A list of list of (word, label) tuples.
    """
    res = []
    for x in X:
        tmp = []
        for word in x:
            if len(word) == 1:
                tmp.append((word, 'S'))
            else:
                tmp.append((word[0], 'B'))
                for w in word[1:-1]:
                    tmp.append((w, 'M'))
                tmp.append((word[-1], 'E'))
        res.append(tmp)
    return res


def get_data():
    X, tmp = [], []
    for ln in cf.io.read('/tmp/segtrain'):
        if not ln:
            if tmp:
                X.append(tmp)
            tmp = []
        else:
            if len(ln) > 2:
                tmp.append((ln[0], ln[2]))

    if tmp:
        X.append(tmp)
    labels = [[p[1] for p in x] for x in X]
    # print(labels[:1])
    return X, labels


def train(X_formatted, labels):
    # X_formatted, labels = get_data()
    cli = CRF(X_formatted, labels)
    cli.fit()
    cli.save_model('/tmp/lcut.model')


def _token(text: str, tags: List[str]) -> List[str]:
    tuples, res = zip(text, tags), []
    for a, b in tuples:
        if b in ('S', 'B'):
            res.append(a)
        else:
            if not res:
                res = ['']
            res[-1] += a
    return res


def test():
    model = CRF.load_model('/tmp/lcut.model')
    raw_texts = [
        '可以做为通用中文语料，做预训练的语料或构建词向量，也可以用于构建知识问答。', '明天天气怎么样?',
        '我爱自由，但是效果看起来很一般。', '全国代表大会高举邓小平理论伟大旗帜',
        '金额类实体识别使用的数据集来源于慧算账公司“询问价位”的事件结果，总计543条数据，大部分是手动标注数据，有一部分人工构造的数据。'
    ]
    cf.info('model loaded')
    texts = [[(w, ) for w in text] for text in raw_texts]
    feature_list = [model._sent2features(text) for text in texts]
    tag_list = model.predict(feature_list)
    for text, tags in zip(raw_texts, tag_list):
        print(_token(text, tags))


# xs = [[x for x in e.split(' ') if x]
#       for e in cf.io.read('/tmp/msr_training.utf8')]
xs = cf.read('/tmp/corpus').top(20000).each(
    lambda x: [e for e in x.split(' ') if e]).filter(len).data
xs = reformat(xs)
print(xs[:10])
# xs, labels = get_data()
labels = [[_[1] for _ in x] for x in xs]
print(xs[:10], labels[:10])
train(xs, labels)

test()
