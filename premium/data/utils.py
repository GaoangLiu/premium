#!/usr/bin/env python3
from abc import ABC, abstractmethod
from enum import Enum
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


class EnumRemoteRepositories(Enum):
    cloudflare = 'https://host.ddot.cc'


class DataRetriver(object):

    def __init__(self, remote: str, local: str, cache_dir: str) -> None:
        self.remote = remote
        self.local = local
        self.cache_dir = os.path.join(cf.io.home() + f'/.cache/{cache_dir}')
        try:
            os.mkdir(self.cache_dir)
        except:
            pass  # already exists
        self._full_path = os.path.join(self.cache_dir, self.local)

    def download(self):
        cf.net.download(self.remote, self._full_path)
        cf.info(f'Downloaded {self.remote} to {self._full_path}')

    @property
    def df(self) -> pd.DataFrame:
        self.download()
        _df = pd.read_csv(self._full_path)
        _df.dropna(inplace=True)
        return _df


class AbstractDataGetter(ABC):
    @abstractmethod
    def exec(self):
        pass

    @abstractmethod
    def get_work_dir(self):
        pass


class ZipDataGetter(AbstractDataGetter):
    def exec(self):
        self.retriver.download()
        self.unpack()

    @abstractmethod
    def unpack(self):
        pass

    def get_work_dir(self):
        return self._work_dir


class CloudflareDataGetter(ZipDataGetter):
    def __init__(self, file_name: str, cache_dir: str = None) -> None:
        super().__init__()
        self.repo_url = EnumRemoteRepositories.cloudflare.value
        self.file_name = file_name
        self.remote_url = cf.urljoin(self.repo_url, self.file_name)
        if cache_dir is None:
            self.cache_dir = self.file_name.split('.')[0]
        else:
            self.cache_dir = cache_dir
        self._work_dir = os.path.join(cf.io.home() + f'/.cache/{self.cache_dir}')
        self.retriver = DataRetriver(self.remote_url, self.file_name, self.cache_dir)

    def exec(self):
        self.retriver.download()
        self.unpack()

    def unpack(self):
        if self.file_name.endswith('zip'):
            os.system(f'unzip {self.retriver._full_path} -d {self.retriver.cache_dir}')
        elif self.file_name.endswith('tar.gz'):
            os.system(f'tar -xvf {self.retriver._full_path} -C {self.retriver.cache_dir}')
        elif self.file_name.endswith('tar.bz2'):
            os.system(f'tar -xvjf {self.retriver._full_path} -C {self.retriver.cache_dir}')
        elif self.file_name.endswith('tar'):
            os.system(f'tar -xvf {self.retriver._full_path} -C {self.retriver.cache_dir}')
        else:
            os.system(f'7z x {self.retriver._full_path} -o{self.retriver.cache_dir}')


class DataGetterFactory(object):
    @classmethod
    def _create_cloudflare_data_getter(cls, file_name: str, cache_dir: str) -> CloudflareDataGetter:
        return CloudflareDataGetter(file_name, cache_dir)

    @classmethod
    def init(cls, file_name: str) -> pd.DataFrame:
        map_ = {'rnn_names': ('pytorch_rnn_names.zip', EnumRemoteRepositories.cloudflare, 'pytorch_rnn_names'),
                'pytorch_rnn_names': ('pytorch_rnn_names.zip', EnumRemoteRepositories.cloudflare, 'pytorch_rnn_names')}
        try:
            file_name, repo, cache_dir = map_[file_name]
            return cls._create_cloudflare_data_getter(file_name, cache_dir)
        except KeyError:
            cf.warning(f'No such file {file_name} in the map')
            return None
