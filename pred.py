#!/usr/bin/env python
import os
import pickle

import codefast as cf
import numpy as np
from rich import print
from tensorflow import keras

from premium.models.nn import MultiClassifier

working_dir = '/tmp/multiclf'
clf = MultiClassifier(max_feature=30000,
                      max_length=200,
                      embedding_dim=100,
                      working_dir=working_dir)


def load_test_data():
    lines = cf.io.read('localdata/test_hema.txt')
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    return lines


def fullpath(filename):
    return os.path.join(working_dir, filename)


args = {
    'tokenizer': pickle.load(open(fullpath('tokenizer.pkl'), 'rb')),
    'model': keras.models.load_model(fullpath('model.h5')),
    'idx2target': cf.js(fullpath('idx2target.json'))
}

clf.load_pretrained(args)

# clf.model = keras.models.load_model(os.path.join(working_dir, 'model.h5'))

# idx2target = cf.js(os.path.join(working_dir, 'idx2target.json'))

test_data = load_test_data()
ypreds = clf.predict(test_data)
ypreds = np.argmax(ypreds, axis=1)
print(ypreds)

ypreds = [clf.idx2target[str(i)] for i in ypreds]
for text, label in zip(test_data, ypreds):
    print(f'{text} => {label}')
