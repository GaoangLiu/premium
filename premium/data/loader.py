#!/usr/bin/env python
from premium.data.utils import make_obj, DataGetterFactory
import os

import codefast as cf
import pandas as pd

try:
    import yaml
except ImportError:
    print('Please install yaml')

from .utils import DataRetriver, Struct, make_obj


def load_yaml(path) -> Struct:
    with open(path) as f:
        return make_obj(yaml.safe_load(f))


def ner_weibo() -> Struct:
    """ ner weibo data
    https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER
    """
    x = DataRetriver('https://host.ddot.cc/weiboNER_2nd_conll.train.csv',
                     'train.csv', 'ner_weibo')
    t = DataRetriver('https://host.ddot.cc/weiboNER_2nd_conll.test.csv',
                     'test.csv', 'ner_weibo')
    v = DataRetriver('https://host.ddot.cc/weiboNER_2nd_conll.dev.csv',
                     'dev.csv', 'ner_weibo')
    return make_obj(dict(train=x.df, test=t.df, val=v.df))


def ner_en() -> Struct:
    """English ner dataset"""
    x = DataRetriver('https://host.ddot.cc/ner_en.csv', 'ner_en.csv', 'ner')
    return make_obj(dict(train=x.df))


def imdb_sentiment() -> Struct:
    """imdb sentiment dataset"""
    x = DataRetriver('https://host.ddot.cc/imdb_sentiment.csv', 'sentiment.csv',
                     'imdb')
    return make_obj(dict(train=x.df))


def spam_en() -> Struct:
    x = DataRetriver('https://host.ddot.cc/spam_en.csv', 'spam_en.csv', 'spam')
    return make_obj(dict(train=x.df))


def loader(dataset_name: str):
    cli = DataGetterFactory.init(dataset_name)
    if cli:
        cli.exec()
        return cli.get_work_dir()
    

