#!/usr/bin/env python
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import numpy as np
import pandas as pd
import tensorflow as tf
from rich import print
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from premium.models.model_config import KerasCallbacks
from premium.preprocessing.text import TextTokenizer
from premium.utils import auto_set_label_num


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


class TextClaissifier(object):
    def __init__(self,
                 data: str,
                 batch_size: int = 32,
                 epochs: int = 10,
                 max_length: int = 200,
                 embedding_size: int = 200,
                 vocab_size: int = 30000,
                 embedding_matrix: np.ndarray = None,
                 num_classes: int = 2,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 verbose: int = 1,
                 working_dir: str = '/tmp/',
                 **kwargs):
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.num_classes = num_classes
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.working_dir = working_dir
        self.kwargs = kwargs
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_size,
                                                   input_length=100,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def build(self):
        raise NotImplementedError

    def _call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x

    @staticmethod
    def load(working_dir: str):
        """ Load pretrained model and tokenizer
        """
        args = {
            'model_path': os.path.join(working_dir, 'model.h5'),
            'tokenizer_path': os.path.join(working_dir, 'tokenizer.pkl'),
            'index_map_path': os.path.join(working_dir, 'index_map.json'),
            'label_map_path': os.path.join(working_dir, 'label_map.json')
        }
        for v in args.values():
            assert os.path.exists(v), '{} does not exist'.format(v)
        clf = TextClaissifier('')
        clf.model = tf.keras.models.load_model(args['model_path'])
        clf.toker = TextTokenizer.load(args['tokenizer_path'])
        clf.label_map = cf.js(args['label_map_path'])
        clf.index_map = cf.js(args['index_map_path'])
        clf.index_map = {int(k): v for k, v in clf.index_map.items()}
        event = {'msg': 'Load pretrained model and tokenizer', 'args': args}
        cf.info(event)
        clf.model.summary()
        return clf

    def fit(self, x, y, epochs: int, batch_size: int, validation_data: Tuple,
            callbacks: KerasCallbacks):
        self.model.fit(x,
                       y,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=validation_data,
                       callbacks=callbacks)
        return self

    def save_model(self, path: str = None):
        path = path or os.path.join(self.working_dir, 'model.h5')
        self.model.save(path)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def to_catorical(self, y: List[Any]) -> np.ndarray:
        """ Convert the labels to categorical, and keep two mapping dicts
        label_map: {label: index}
        index_map: {index: label}
        """
        self.label_map, ys = auto_set_label_num(y)
        self.index_map = {v: k for k, v in self.label_map.items()}
        cf.js.write(self.label_map,
                    os.path.join(self.working_dir, 'label_map.json'))
        cf.js.write(self.index_map,
                    os.path.join(self.working_dir, 'index_map.json'))
        event = {
            'label_map': self.label_map,
            'index_map': self.index_map,
            'msg': 'label map exported to {}'.format(self.working_dir)
        }
        cf.info(event)
        return tf.keras.utils.to_categorical(ys, num_classes=self.num_classes)


class Trainer(TextClaissifier):
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
    def fit(self):
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
        super().fit(X_train,
                    y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(X_test, y_test),
                    callbacks=KerasCallbacks().some(
                        ['reduce_lr', 'early_stopping'])).save_model()
        self.toker.save()
        return self


def train():
    args = {
        'data': 'localdata/train_hemabot.csv',
        'batch_size': 32,
        'epochs': 5,
        'working_dir': '/tmp/hema/',
        'num_classes': 6,
    }
    trainer = Trainer(**args)
    trainer.fit()


if True:
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    clf = Trainer.load('/tmp/hema')

    @app.route('/', methods=['POST'])
    def predict():
        data = request.get_json()
        text = data['text']
        if text == 'exit':
            return
        ipt = pd.DataFrame({'text': [text]})
        ipt = feature_engineer(ipt)
        try:
            pred = clf.predict([
                clf.toker.tok(ipt.text), ipt.has_video,
                ipt.has_leading_oncemessage, ipt.has_twitter
            ])
            pred = np.argmax(pred, axis=1)
            return jsonify({
                'label': clf.index_map[pred[0]],
            })
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)})

    app.run(host='0.0.0.0', port=5001)
