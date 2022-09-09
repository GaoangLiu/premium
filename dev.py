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
import pandas as pd
import tensorflow as tf
from codefast.decorators.log import time_it
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers

ds = load_dataset('conll2003')
pos_tag_list = []
ner_tag_list = []
token_list = []
for row in ds['train']:
    pos_tag_list.append(row['pos_tags'])
    ner_tag_list.append(row['ner_tags'])
    token_list.append(' '.join(row['tokens']))

num_tags = 9

model = keras.Sequential([
    keras.layers.TextVectorization(output_mode='int'),
    keras.layers.Embedding(input_dim=1000, output_dim=64),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_tags, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.build(input_shape=(None, 128))
model.summary()
history = model.fit(token_list, ner_tag_list, epochs=10)
