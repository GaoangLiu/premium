#!/usr/bin/env python
import codefast as cf
from typing import List, Dict, Tuple, Set, Optional
import tensorflow as tf
import premium as pm
import numpy as np
import pandas as pd
from dofast import SyncFile
from dofast import SyncFile


class KaggleData(object):
    """ Generally, three types of data are loaded:
    train.csv
    test.csv
    sample_submission.csv
    """
    def __init__(self,
                 local_dir: str,
                 remote_dir: str,
                 loader_name='filedn_loader') -> None:
        self.x = SyncFile('train.csv',
                          remote_dir=remote_dir,
                          local_dir=local_dir,
                          loader_name=loader_name)
        self.xt = SyncFile('test.csv',
                           remote_dir=remote_dir,
                           local_dir=local_dir,
                           loader_name=loader_name)
        self.sub = self.x.clone('sample_submission.csv')

    def standard_load(self) -> Tuple[pd.DataFrame]:
        x = self.x.read_csv().df
        xt = self.xt.read_csv().df
        sub = self.sub.read_csv().df
        cf.info('train shape is {}'.format(x.shape))
        cf.info('train head is {}'.format(x.head()))
        cf.info('train random sample is {}'.format(x.sample(5)))
        cf.info('columns of train is {}'.format(x.columns))
        cf.info('test shape is {}'.format(xt.shape))
        cf.info('data load success ✅')
        return x, xt, sub
