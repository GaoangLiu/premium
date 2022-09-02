#!/usr/bin/env python
from typing import Callable, Dict, List, Set, Tuple, Union

import codefast as cf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Dense, Embedding, GlobalAveragePooling1D,
                                     TextVectorization)
from tensorflow.keras.models import Sequential

try:
    import fasttext
except ImportError as e:
    cf.error("import fasttext failed", e)


def myfastText(df: pd.DataFrame,
               embedding_dims: int = 100,
               ngrams: int = 2,
               max_features: int = 30000,
               maxlen: int = 400,
               batch_size: int = 32,
               epochs: int = 10):
    """ A simple implementation of fastText.
    """
    X, Xv, y, yv = train_test_split(df['text'], df['target'], random_state=0)
    args = {
        'dim': embedding_dims,
        'ngrams': ngrams,
        'max_features': max_features,
        'maxlen': maxlen,
        'batch_size': batch_size,
        'epochs': epochs
    }
    cf.info(args)
    vectorize_layer = TextVectorization(
        ngrams=ngrams,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=maxlen,
    )
    vectorize_layer.adapt(X.values)
    model = Sequential([
        vectorize_layer,
        Embedding(max_features + 1, embedding_dims),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xv, yv),
    )
    return model, history


def check_pretrained_vector(vector_path: str) -> bool:
    """<pretrainedVectors> file must starts with a line contains the number of
    words in the vocabulary and the size of the vectors. E.g., 
    100000 200
    Refer: https://fasttext.cc/docs/en/english-vectors.html
    """
    with open(vector_path, 'r') as f:
        first_line = f.readline().strip()
        if ' ' not in first_line:
            raise InvalidVectorException('Invalid vector file')
        num_words, dim = first_line.split(' ')
        if not num_words.isdigit() or not dim.isdigit():
            raise InvalidVectorException('Invalid vector file')
        return True


def split_data(df: pd.DataFrame):
    assert 'text' in df.columns, 'text column not found'
    assert 'target' in df.columns, 'target column not found'
    df['target'] = '__label__' + df['target'].astype(str)
    df['text'] = df['text'].astype(str)
    df['label'] = df['target']
    msg = {'label_count': df['label'].value_counts()}
    cf.info(msg)

    fasttext_input = df[['target', 'text']].astype(str)
    size = int(fasttext_input.shape[0]*0.8)
    fasttext_train = fasttext_input.sample(size, random_state=0)
    fasttext_valid = fasttext_input.drop(fasttext_train.index)
    cf.info('Train data size: {}'.format(len(fasttext_train)))
    cf.info('Valid data size: {}'.format(len(fasttext_valid)))

    fasttext_train.to_csv("/tmp/tt.train",
                          quotechar=" ",
                          header=False,
                          index=False)
    fasttext_valid.to_csv("/tmp/tt.test",
                          quotechar=" ",
                          header=False,
                          index=False)
    return fasttext_train, fasttext_valid


def baseline(df: pd.DataFrame,
             dim: int = 200,
             pretrainedVectors: str = None,
             model_path: str = None,
             deprecate_split:bool=False,
             *args):
    """ 
    Inputs:
        deprecate_split: bool, do not split data again if True. 
    """
    model_path = '/tmp/pyfasttext.bin' if not model_path else model_path
    if not deprecate_split:
        _, _ = split_data(df)
    cf.info('start training')
    train_args = {
        'input': '/tmp/tt.train',
        'dim': dim,
        'thread': 12,
    }

    if pretrainedVectors:
        check_pretrained_vector(pretrainedVectors)
        train_args['pretrainedVectors'] = pretrainedVectors
    model = fasttext.train_supervised(**train_args)
    model.save_model(model_path)

    # validate the model
    res = model.test("/tmp/tt.test")
    cf.info('validate result', res)
    return model


def autotune(df: pd.DataFrame,
             dim: int = 200,
             pretrainedVectors: str = None,
             model_path: str = None,
             autotuneDuration:float=300,
             *args):
    # Find the best possible hyperparameters
    model_path = '/tmp/pyfasttext.bin' if not model_path else model_path
    _, _ = split_data(df)
    cf.info('start training')
    train_args = {
        'input': '/tmp/tt.train',
        'dim': dim,
        'thread': 12,
        'autotuneValidationFile': '/tmp/tt.test',
        'autoTuneDuration': autotuneDuration,
    }

    if pretrainedVectors:
        check_pretrained_vector(pretrainedVectors)
        train_args['pretrainedVectors'] = pretrainedVectors
    cf.info('Auto tune started...')
    model = fasttext.train_supervised(**train_args)
    model.save_model(model_path)
    return model


def train_vector(df: pd.DataFrame, *kargs, **kwargs):
    pass


if __name__ == '__main__':
    df = pd.read_csv('/tmp/imdb_sentiment.csv')
    df = df.sample(frac=0.5)
    df['text'] = df['review']
    df['target'] = df['sentiment']
    myfastText(df)
