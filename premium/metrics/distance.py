#!/usr/bin/env python
import random
import re, os, sys, joblib
from collections import defaultdict
from functools import reduce
import codefast as cf
from typing import List, Dict, Tuple, Set, Union, Optional, Any


def jaccard(a: Set, b: Set) -> float:
    return len(a & b) / len(a | b)


def wmd(s1: List, s2: List, model: 'gensim.downloader'=None) -> float:
    """Word Mover's Distance. 
    model can be gensim.downloader or gensim.models.KeyedVectors

    Example 1: 
    >>> import gensim.downloader as api
    >>> model = api.load('word2vec-google-news-300')
    >>> model.wmdistance(['hello', 'world'], ['hello', 'world'])

    Example 2:    
    >>> model_path = '/data/pretraining/tencent100.txt'
    >>> model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    >>> model.wmdistance(s1, s2)
    """
    dis = model.wmdistance(s1, s2)
    return dis
