#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from itertools import chain
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import joblib
import nltk
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn_crfsuite
from rich import print
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics, scorers


class CRF(object):
    """ A skleanr crfsuite wrapper for fast model training and deployment
    """
    def __init__(self, X: List, y: List, feature_template: str = None) -> None:
        """
        Args:
            X: input data
            y: labels
            feature_template: path of feature template file, follow https://taku910.github.io/crfpp/ for template example
        """
        self.X = X
        self.y = y
        self.feature_template = feature_template
        self.is_feature_extracted = False
        self.model = None

    def _word2features(self, s: List[Tuple],
                       i: int) -> Dict[str, Union[str, List[str]]]:
        """ Convert a word to features. 
        Usually, we don't have much information on samples, so we only use words
        as features. But if we do have extra information, such as POS tag, NER tag, etc,
        then we can leverage them to improve the performance by adding features
        such as 'U[-1]_POS': s[i-1][1], which means the POS tag of the previous word.
        """
        return {
            'U[0]':
            s[i][0],
            'U[-1]':
            s[i - 1][0] if i > 0 else '<START>',
            'U[-2]':
            s[i - 2][0] if i > 1 else '<START>',
            'U[+1]':
            s[i + 1][0] if i < len(s) - 1 else '<END>',
            'U[+2]':
            s[i + 2][0] if i < len(s) - 2 else '<END>',
            'B[-1]': [s[i - 1][0], s[i][0]] if i > 0 else '<START>',
            'B[+1]': [s[i][0], s[i + 1][0]] if i < len(s) - 1 else '<END>',
            'B[-1/1]': [s[i - 1][0], s[i][0], s[i + 1][0]]
            if i > 0 and i < len(s) - 1 else '<START_OR_END>',
        }

    def _sent2features(self, s: List):
        """ Convert a sentence to features
        """
        return [self._word2features(s, i) for i, _ in enumerate(s)]

    def extract_features(self):
        """ Extract features from input data
        """

        self.X = [self._sent2features(s) for s in self.X]
        self.is_feature_extracted = True
        return self.X

    def fit(self):
        if not self.is_feature_extracted:
            self.X = self.extract_features()
        self.model = sklearn_crfsuite.CRF(algorithm='lbfgs',
                                          c1=0.1,
                                          c2=0.1,
                                          epsilon=0.01,
                                          max_iterations=300,
                                          verbose=True,
                                          all_possible_transitions=True)
        cf.info('crf model created')
        self.model.fit(self.X, self.y)

    @classmethod
    def load_model(cls, model_path: str)->'CRF':
        return pickle.load(open(model_path, 'rb'))

    def save_model(self, model_path: str):
        pickle.dump(self, open(model_path, 'wb'))
        return True

    def predict(self, X: List):
        """ Predict labels for input data
        """
        return self.model.predict(X)

    def evaluate(self):
        pass

    def predict_proba(self):
        pass


def get_data(datafile: str) -> List[List[Tuple[str, str]]]:
    """ Convert data to <X, y> tuples. 
    """
    chunks = []
    for ln in cf.io.read(datafile):
        if not ln:
            chunks.append([])
        else:
            if not chunks:
                chunks.append([])
            chunks[-1].append((ln[0], ln[2]))
    return chunks


def sent2labels(inputs: List[Tuple]):
    return [label for _, label in inputs]


train_sents = get_data('/tmp/money_train.txt')
test_sents = get_data('/tmp/money_test.txt')

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Train new CRF model")
parser.add_argument("--test", help="Test CRF model")

args = parser.parse_args()

y_train = [sent2labels(s) for s in train_sents]
X_train = train_sents
mycrf = CRF(X_train, y_train)
print(X_train[0], y_train[0])

X_test = test_sents
y_test = [sent2labels(s) for s in test_sents]

if args.train:
    mycrf.fit()
    mycrf.save_model('crf.model')


class MoneyEntity(object):
    _MoneyTag = 'M'
    _OtherTag = 'O'

    def __init__(self, text: str, label: Union[List[str], str]) -> None:
        self.text = text
        self.label = label
        self.combo = list(zip(self.text, self.label))

    def values(self) -> List[str]:
        indexes = []
        size, a = len(self.label), 0
        for i, l in enumerate(self.label):
            if l != self._MoneyTag:
                continue
            if i == 0 or (i > 0 and self.label[i - 1] == self._OtherTag):
                a = i
            elif i == size - 1 or (i < size - 1
                                   and self.label[i + 1] == self._OtherTag):
                indexes.append((a, i))
        return [self.text[a:b + 1] for a, b in indexes]

    def __repr__(self) -> str:
        pass


def ner(crf:CRF, text: str) -> List[str]:
    token_list = [(w, ) for w in list(text)]
    feature_list = [crf._sent2features(token_list)]
    y_pred = crf.predict(feature_list)
    return y_pred[0]

if args.test:
    crf = CRF.load_model(args.test)
    cf.info('model loaded')
    texts = [
        '我想买一张北京到上海的机票，价格差不多三千块钱，觉得3743块的机票不错',
        '没听明白，是这样子的，就比如说我们的工资，一个月是2300块钱吗。', '平均下来一个月的话就是400多块钱一个月吗。',
        '我这边只有三百块，加上你的有1200块，加上昨天的七千六百四十三元，不知道够不够',
        '2000多，然后2000加上一个地址，反正也就6000多块钱吧，好像。',
        '然后现在两年的话，它是有活动价9988，平均下来一个月的话就是400多块钱一个月吗。',
        '如果说是免这免免那个个人所得税额度是在6万，现在是每个人对每个人的那个年度所得，就是每个月5000嘛，对不对，一年就是6万块吗？超了6万块之后，超出额度是按七级去缴纳它的一个个人所得税的。',
        '没有的话，现在618出了一个特定的套餐，比之前给您报的那个价格便宜。', '我们一年是2000吗', '你两年就是三千九百九块一年',
        '你两年就是2347美元一年', '价格是三千九百九一年'
    ]
    for text in texts:
        y_pred = ner(crf, text)
        entity = MoneyEntity(text, y_pred)
        print(entity.values())
    # y_pred = crf.predict(X_test[:1])
    # print(y_pred)

    # for i, y in enumerate(y_pred):
    #     tp = zip(y, test_sents[i])
    #     for pred, label in tp:
    #         pred = pred.strip()
    #         print([label, pred])

    # resp = metrics.flat_f1_score(y_test, y_pred, average='weighted')
    # print(resp)
