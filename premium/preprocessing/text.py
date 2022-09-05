#!/usr/bin/env python
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import tensorflow as tf
from codefast.decorators.log import time_it
from tensorflow import keras

class TextVectorizer(tf.keras.layers.TextVectorization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adapt(self, data):
        super().adapt(data)
        self.vocab = self.get_vocabulary()
        self.word_index = dict(zip(self.vocab, range(len(self.vocab))))
    
    @time_it
    def dump(self, path: str):
        self.pickle(path)

    @time_it
    def pickle(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'config': self.get_config(),
                    'weights': self.get_weights(),
                }, f)

    @classmethod
    @time_it
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(**data['config'])
        obj.set_weights(data['weights'])
        return obj


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

    @time_it
    def fit_transform(self, texts: List[str]) -> List[List[int]]:
        """ Fit the tokenizer on the texts 
        """
        super().fit_on_texts(texts)
        return self.tok(texts)
