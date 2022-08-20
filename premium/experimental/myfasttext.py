#!/usr/bin/env python
import codefast as cf
import fasttext
import numpy as np
import pandas as pd


class InvalidVectorException(Exception):
    pass


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


def benchmark(df: pd.DataFrame, dim: int = 200, pretrainedVectors: str = None, model_path: str = None, *args):
    assert 'text' in df.columns, 'text column not found'
    assert 'target' in df.columns, 'target column not found'
    df['target'] = '__label__' + df['target'].astype(str)
    df['text'] = df['text'].astype(str)
    df['label'] = df['target']
    model_path = '/tmp/pyfasttext.bin' if not model_path else model_path
    msg = {'label_count': df['label'].value_counts(), 'model_path': model_path}
    cf.info(msg)

    fasttext_input = df[['target', 'text']].astype(str)
    np.random.seed(0)
    msk = np.random.rand(len(df)) < 0.8
    fasttext_train = fasttext_input[msk]
    fasttext_valid = fasttext_input[~msk]
    cf.info('Train data: {}'.format(len(fasttext_train)))
    cf.info('Valid data: {}'.format(len(fasttext_valid)))

    fasttext_train.to_csv("/tmp/tt.train",
                          sep=" ",
                          quotechar=" ",
                          header=False,
                          index=False)
    fasttext_valid.to_csv("/tmp/tt.test",
                          sep=" ",
                          quotechar=" ",
                          header=False,
                          index=False)

    cf.info('start training')
    train_args = {'input': '/tmp/tt.train', 'dim': dim,
                  'minn': 1, 'maxn': 7, 'thread': 12}

    if pretrainedVectors:
        check_pretrained_vector(pretrainedVectors)
        train_args['pretrainedVectors'] = pretrainedVectors
    model = fasttext.train_supervised(**train_args)
    model.save_model(model_path)

    # validate the model
    res = model.test("/tmp/tt.test")
    cf.info('validate result', res)
    return model
