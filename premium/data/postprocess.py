#!/usr/bin/env python
import codefast as cf
import numpy as np
import pandas as pd

import scipy
from typing import List
from argparse import Namespace


def mode(A: np.ndarray, compress: str = 'row') -> np.ndarray:
    """ Returns an array of the most common element in each row or column.
    Args:
        A: 2D array
        compress: 'row' or 'col', compress the array along the specified axis. 
        If compress is 'row', the array is compressed along the rows. E.g., 
        A = [[1, 2, 3], [2, 2, 3], [3, 2, 3]] -> [[1], [2], [3]]
        If compress is 'col', the array is compressed along the columns. E.g.,
        A = [[2, 1, 3], 
             [2, 2, 3], 
             [3, 1, 3]] -> [[2, 1, 3]]
    """
    axis = 1 if compress == 'row' else 0
    return scipy.stats.mode(A, axis=axis)[0]


array = Namespace(mode=mode)


class Mop(object):
    def hard_vote(self,
                  files: List[str],
                  target_dtype: str = 'int',
                  export_to: str = None) -> 'Mop':
        dfs = [pd.read_csv(f) for f in files]
        cf.info('blending files {}'.format(files))
        target_name = dfs[-1].columns[-1]
        cf.info('target_name: {}'.format(target_name))
        res = [df[target_name].values for df in dfs]
        lastdf = dfs[-1]
        res = mode(np.array(res), compress='col').flatten()
        lastdf[target_name] = res
        lastdf[target_name] = lastdf[target_name].astype(target_dtype)
        if export_to:
            cf.info('Export to {}'.format(export_to))
            lastdf.to_csv(export_to, index=False)
        return lastdf


mop = Mop()


def get_binary_prediction(y_pred: list, threshold: float = 0.5):
    cf.info(f'Get binary prediction of y_pred')
    ans = []
    for e in y_pred:
        if isinstance(e, int) or isinstance(e, float):
            assert 0 <= e <= 1
            ans.append(1 if e >= threshold else 0)
        elif isinstance(e, list) or isinstance(e, np.ndarray):
            assert len(e) == 1, 'item should contains only one number.'
            n_float = float(e[0])
            assert 0 <= n_float <= 1
            ans.append(1 if n_float >= threshold else 0)
        else:
            print(e, type(e))
            raise TypeError('Unsupported element type.')
    return ans
