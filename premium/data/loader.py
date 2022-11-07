#!/usr/bin/env python
import pandas as pd
import os
import codefast as cf
try:
    import yaml
except ImportError:
    print('Please install yaml')


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self) -> str:
        _dict = {}
        for k, v in self.__dict__.items():
            _dict[k] = v.__dict__ if isinstance(v, Struct) else v
        return str(_dict)

    def __getitem__(self, key):
        return self.__dict__[key]


def make_obj(obj):
    if isinstance(obj, dict):
        _struct = Struct()
        for k, v in obj.items():
            if isinstance(v, dict) or isinstance(v, list):
                _struct.__dict__[k] = make_obj(v)
            else:
                _struct.__dict__[k] = v
        return _struct
    elif isinstance(obj, list):
        return [make_obj(o) for o in obj]
    else:
        return obj


def load_yaml(path) -> Struct:
    with open(path) as f:
        return make_obj(yaml.safe_load(f))


class _Cache(object):
    def __init__(self, remote: str, local: str, cache_dir: str) -> None:
        self.remote = remote
        self.local = local
        self.cache_dir = os.path.join(cf.io.home() + f'/.cache/{cache_dir}')
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self._full_path = os.path.join(self.cache_dir, self.local)

    @property
    def df(self) -> pd.DataFrame:
        cf.net.download(self.remote, self._full_path)
        _df = pd.read_csv(self._full_path)
        _df.dropna(inplace=True)
        return _df


def ner_weibo() -> Struct:
    """ ner weibo data
    https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER
    """
    x = _Cache(
        'https://host.ddot.cc/weiboNER_2nd_conll.train.csv', 'train.csv', 'ner_weibo')
    t = _Cache('https://host.ddot.cc/weiboNER_2nd_conll.test.csv',
               'test.csv', 'ner_weibo')
    v = _Cache('https://host.ddot.cc/weiboNER_2nd_conll.dev.csv',
               'dev.csv', 'ner_weibo')
    return make_obj(dict(train=x.df, test=t.df, val=v.df))


def ner_en() -> Struct:
    """English ner dataset"""
    x = _Cache(
        'https://host.ddot.cc/ner_en.csv', 'ner_en.csv', 'ner')
    return make_obj(dict(train=x.df))
