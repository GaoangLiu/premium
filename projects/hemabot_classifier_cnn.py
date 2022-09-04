#!/usr/bin/env python
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from premium.models.model_config import KerasCallbacks
from premium.preprocessing.text import TextTokenizer
from premium.models.nn import TextClaissifier


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


def has_video_feature(text) -> bool:
    video_format = [
        'mp4', 'flv', 'avi', 'rmvb', 'wmv', 'mkv', 'mov', 'mpg', 'mpeg'
    ]
    for fmt in video_format:
        if text.find(fmt) != -1:
            return True
    return False


def has_twitter_feature(text) -> bool:
    return text.find('twitter') != -1


def has_leading_oncemessage(text) -> bool:
    keywords = ['oncemessage', 'oncemsg', 'onecemesage', 'oncemassage']
    for keyword in keywords:
        if text.startswith(keyword):
            return True
    return False


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    def process_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text

    df['text'] = df['text'].apply(process_text)
    df['has_video'] = df.text.apply(has_video_feature)
    df['has_leading_oncemessage'] = df.text.apply(has_leading_oncemessage)
    df['has_twitter'] = df.text.apply(has_twitter_feature)
    return df


class BotClassifier(TextClaissifier):
    def __init__(self, data: str, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.model = self.build()

    def build(self):
        text_input = tf.keras.layers.Input(shape=(self.max_length, ),
                                           dtype=tf.int32)
        embed = tf.keras.layers.Embedding(self.vocab_size, 128)(text_input)
        conv = tf.keras.layers.Conv1D(5, 3, activation='relu')(embed)
        pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
        video = tf.keras.layers.Input(shape=(1, ), dtype='float32')
        leading = tf.keras.layers.Input(shape=(1, ), dtype='float32')
        twitter = tf.keras.layers.Input(shape=(1, ), dtype='float32')
        x = tf.keras.layers.concatenate([pool, video, leading])
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=[text_input, video, leading, twitter], outputs=x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    @time_decorator
    def call(self):
        df = pd.read_csv(self.data)
        df = feature_engineer(df)
        X = df[['text', 'has_video', 'has_leading_oncemessage', 'has_twitter']]
        y = df['target']
        y = self.to_catorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.1)
        self.toker = TextTokenizer(maxlen=self.max_length,
                                   padding='pre',
                                   path=os.path.join(self.working_dir,
                                                     'tokenizer.pkl'))
        X_train = [
            self.toker.fit_transform(X_train['text']), X_train['has_video'],
            X_train['has_leading_oncemessage'], X_train['has_twitter']
        ]
        X_test = [
            self.toker.tok(X_test['text']), X_test['has_video'],
            X_test['has_leading_oncemessage'], X_test['has_twitter']
        ]
        super().call(X_train,
                     y_train,
                     batch_size=self.batch_size,
                     epochs=self.epochs,
                     validation_data=(X_test, y_test),
                     callbacks=KerasCallbacks().some(
                         ['reduce_lr', 'early_stopping'])).save_model()
        self.toker.save()
        return self


from flask import Flask, jsonify, request

app = Flask(__name__)

clf = BotClassifier.load('/tmp/hema')


@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    ipt = pd.DataFrame({'text': [text]})
    ipt = feature_engineer(ipt)
    try:
        pred = clf.predict([
            clf.toker.tok(ipt.text), ipt.has_video,
            ipt.has_leading_oncemessage, ipt.has_twitter
        ])
        pred = np.argmax(pred, axis=1)
        pred = clf.index_map[pred[0]]
        event = {'input text': text, 'prediction': pred}
        cf.info(event)
        return jsonify({'label': pred})
    except Exception as e:
        cf.error(e)
        return jsonify({'error': str(e)})


app.run(host='0.0.0.0', port=5000)
