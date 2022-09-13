#!/usr/bin/env python
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from timeit import timeit
from typing import Dict, List, Optional, Set, Tuple

import codefast as cf
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from codefast.decorators.log import time_it
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# refer:https://github.com/Akshayc1/named-entity-recognition/blob/master/NER%20using%20Bidirectional%20LSTM%20-%20CRF%20.ipynb

ds = load_dataset('conll2003')
labels = []
texts = []
print(ds['train'][0])
for row in ds['train']:
    labels.append(row['ner_tags'])
    texts.append(' '.join(row['tokens']))

unique_ner_number = len(set([item for sublist in labels for item in sublist]))
info = {'unique_ner_number': unique_ner_number}
cf.info(info)
vocab_size = 10000
max_len = 128

text_vectorizer = keras.layers.TextVectorization(
    max_tokens=vocab_size, output_sequence_length=max_len)

text_vectorizer.adapt(texts)
X_train = text_vectorizer(texts).numpy()
y = keras.preprocessing.sequence.pad_sequences(maxlen=max_len,
                                               sequences=labels,
                                               padding="post",
                                               value=-1)

y = [
    keras.utils.to_categorical(i, num_classes=unique_ner_number + 1) for i in y
]
# y_train.reshape(-1, y_train.shape[-1])

info = {
    'x shape': X_train.shape,
    'y shape': np.array(y).shape,
    'x[0]': X_train[0],
    'y[0]': y[0]
}
cf.info(info)

model = keras.Sequential([
    keras.Input(shape=(max_len, )),
    keras.layers.Embedding(input_dim=vocab_size,
                           output_dim=50,
                           input_length=max_len,
                           mask_zero=True),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(unique_ner_number + 1, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y, epochs=1)
