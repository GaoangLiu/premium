#!/usr/bin/env python
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import tensorflow as tf

def time_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        cf.info('Calling function: {}'.format(func.__name__))
        res = func(*args, **kwargs)
        end = time.time()
        cf.info('Function {} took {:<.4} seconds'.format(
            func.__name__, end - start))
        return res

    return wrapper


class TextTokenizer(tf.keras.preprocessing.text.Tokenizer):
    def __init__(self,
                 maxlen: int,
                 path: str = None,
                 padding: str = 'pre',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.maxlen = maxlen
        self.padding = padding

    def save(self):
        if self.path is None:
            raise ValueError('Path is not set')
        with open(self.path, 'wb') as f:
            pickle.dump(self, f)

    def fit(self, texts: List[str]):
        """ Fit the tokenizer on the texts 
        """
        super().fit_on_texts(texts)
        return self.tok(texts)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def tok(self, texts: List[str]) -> List[List[int]]:
        sequences = self.texts_to_sequences(texts)
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.maxlen, padding=self.padding)

    def transform(self, texts: List[str]) -> List[List[int]]:
        """alias of tok"""
        return self.tok(texts)

    @time_decorator
    def fit_transform(self, texts: List[str]) -> List[List[int]]:
        """ Fit the tokenizer on the texts 
        """
        super().fit_on_texts(texts)
        return self.tok(texts)
