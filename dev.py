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
from tensorflow import keras
from tensorflow.keras import layers

from premium.preprocessing.text import TextVectorizer

df = pd.read_csv('/tmp/imdb_sentiment.csv')
myvec = TextVectorizer(max_tokens=10000,
                       output_mode='int',
                       output_sequence_length=100)
myvec.adapt(df['text'].values)
xs = myvec('hello world')
print(xs)
myvec.pickle('/tmp/myvec.pickle')

myvec2 = TextVectorizer.load('/tmp/myvec.pickle')
xs2 = myvec2('hello world')
print(xs2)
exit(0)

vectorizer = layers.TextVectorization(max_tokens=20000,
                                      output_sequence_length=100)
df = df.sample(100)
vectorizer.adapt(df['text'].values)
xs = vectorizer('this')
print(xs)
# Pickle the config and weights
pickle.dump(
    {
        'config': vectorizer.get_config(),
        'weights': vectorizer.get_weights()
    }, open("tv_layer.pkl", "wb"))

print("*" * 10)
# Later you can unpickle and use
# `config` to create object and
# `weights` to load the trained weights.

from_disk = pickle.load(open("tv_layer.pkl", "rb"))
new_v = layers.TextVectorization.from_config(from_disk['config'])
new_v.set_weights(from_disk['weights'])

# Lets see the Vector for word "this"
print(new_v("this"))
print('Done')
