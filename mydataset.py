#!/usr/bin/env python
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce

import codefast as cf
import jieba
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import torch as T
from rich import print
from sklearn.model_selection import train_test_split
from tensorflow import keras

fp = '/tmp/imdb_sentiment.csv'
fp = '/tmp/waimai_10k.csv'
# fp = '/tmp/online_shopping_10_cats.csv'
fp = 'x.csv'
df = pd.read_csv(fp)
df = df.dropna(axis=0, how='any')
df = df[df['review'].apply(lambda x: len(x) > 10)]


def to_chars(text: str):
    """ Split Chinese to characters
    """
    return tf.strings.unicode_split(text,
                                    input_encoding='UTF-8',
                                    errors="ignore")


# df['review'] = df.review.apply(lambda x: ' '.join(jieba.cut(x)))
train, test = train_test_split(df)
# train_dataset = tf.data.Dataset.from_tensor_slices(train.review)
# train.to_dict(orient='list'))
max_features = 10000     # Maximum vocab size.
max_len = 50     # Sequence length to pad the outputs to.

# Create the layer.


def custom_standardization(input_data):
    return tf.strings.regex_replace('/'.join(jieba.lcut(input_data)), '/', ' ')


vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_len,
    split='character')

# split='whitespace')
vectorize_layer.adapt(train.review)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


print(random.sample(vectorize_layer.get_vocabulary(), 10))

xs = vectorize_layer(['每次点的都够吃两次', '味道正宗，量大内容多'])
print(xs)
