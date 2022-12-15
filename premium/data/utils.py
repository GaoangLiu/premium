#!/usr/bin/env python3
import json
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import joblib
import numpy as np
import pandas as pd
from rich import print


class Struct(object):

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


class DataRetriver(object):

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
