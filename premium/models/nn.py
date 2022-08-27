#!/usr/bin/env python
import random
from typing import Callable, Dict, List, Tuple, Union

import codefast as cf
import numpy as np  # linear algebra
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout,
                                     Embedding, GlobalMaxPool1D, Input)
from tensorflow.keras.preprocessing.sequence import pad_sequences

import premium as pm
from premium.data.postprocess import get_binary_prediction
from premium.data.preprocess import pad_sequences, tokenize
from premium.models.model_config import KerasCallbacks
from premium.utils import auto_set_label_num


def to_chars(text: str):
    """For splitting the input text, expecially non-English text. 
    refer https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
    for more detail
    """
    return tf.strings.unicode_split(text,
                                    input_encoding='UTF-8',
                                    errors="ignore")


class BinClassifier(object):
    """ Binary LSTM classifier
    """

    def __init__(
        self,
        max_feature: int = 20000,
        max_length: int = 100,
        embedding_dim: int = 200,
        pretrained_vector_path: str = None,
        vectorizer_split_strategy: Union[Callable,
                                         str] = 'whitespace') -> None:
        """ LSTM Binary classifcation model
        Inputs: 
            vectororizer_split_strategy: split stratergy. Use whitespace for English and
                charactor for Chinese.
        """
        self.max_feature = max_feature
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.keras_callbacks = KerasCallbacks()
        self.vectorizer_split_strategy = vectorizer_split_strategy
        self.prerained_vector_path = pretrained_vector_path
        args = {
            'max_length': self.max_length,
            'embeding_dim': self.embedding_dim,
            'max_feature': self.max_feature,
            'vectorizer_split_strategy': self.vectorizer_split_strategy,
            'prerained_vector_path': self.prerained_vector_path
        }
        cf.info('args: {}'.format(args))

    def vectorize(self, df: pd.DataFrame) -> Tuple:
        """Split dataset into (train, test). Vectorize on train corpus.
        """
        df.dropna(inplace=True)
        df_train, df_test = train_test_split(df)

        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=self.max_feature,
            output_mode='int',
            output_sequence_length=self.max_length,
            split=self.vectorizer_split_strategy)
        vectorize_layer.adapt(df_train.text)
        event = {
            'msg': "vectorizing completes",
            "samples": random.sample(vectorize_layer.get_vocabulary(), 10)
        }
        cf.info(event)
        return (vectorize_layer(df_train.text),
                df_train.target), (vectorize_layer(df_test.text),
                                   df_test.target)

    def tokenize(self, texts: List[str]) -> Tuple:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(texts)
        text_sequences = tokenizer.texts_to_sequences(texts)
        text_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            text_sequences, padding='pre', maxlen=self.max_length)
        self.vocab_size = len(tokenizer.word_index) + 1
        event = {
            'msg': "tokenizing completes",
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        cf.info(event)
        return text_sequences, tokenizer

    def get_embed_matrix(self,
                         tokenizer: tf.keras.preprocessing.text.Tokenizer, pretrained_vector_path: str):
        """Get embedding matrix from tokenizer
        """
        from gensim.models import KeyedVectors
        vectors = KeyedVectors.load_word2vec_format(pretrained_vector_path)
        vocab_size = len(tokenizer.word_index) + 1
        weight_matrix = np.zeros((vocab_size, self.embedding_dim))
        for word, i in tokenizer.word_index.items():
            try:
                weight_matrix[i] = vectors[word]
            except KeyError:
                weight_matrix[i] = np.random.uniform(-5, 5, self.embedding_dim)
        return weight_matrix

    def build_model(self, embed_matrix: np.ndarray = None) -> tf.keras.Model:
        sentence_input = tf.keras.layers.Input(shape=(self.max_length, ))
        x = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_dim,
            input_length=self.max_length)(sentence_input)
        if embed_matrix is not None:
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim,
                                          weights=[embed_matrix],
                                          input_length=self.max_length,
                                          trainable=False)(sentence_input)
        x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.LSTM(100)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(100, activation='selu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(sentence_input, output)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        cf.info('model compiled')
        print(model.summary())
        return model

    def fit(self,
            train_ds: Tuple,
            test_ds: Tuple,
            model,
            batch_size: int = 32,
            epochs: int = 5):
        callbacks = self.keras_callbacks.some(
            ['early_stopping', 'reduce_lr', 'csv_logger'])

        history = model.fit(train_ds[0],
                            train_ds[1],
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=test_ds,
                            callbacks=callbacks,
                            use_multiprocessing=True)

        return history, model

    def predict(self, X_te):
        return self.model.predict(X_te,
                                  batch_size=1024,
                                  verbose=1,
                                  use_multiprocessing=True)

    def benchmark(self,
                  df: pd.DataFrame,
                  batch_size: int = 32,
                  epochs: int = 3):
        """Do a quick benchmark on given dataset in format
        target, text
        """
        assert 'target' in df.columns, 'target must be in columns'
        assert 'text' in df.columns, 'text must be in columns'
        X, tokenizer = self.tokenize(df.text)
        if self.prerained_vector_path:
            embed_matrix = self.get_embed_matrix(tokenizer, self.prerained_vector_path)
            model = self.build_model(embed_matrix)
        else:
            model = self.build_model()
        history = model.fit(X,
                            df.target,
                            validation_split=0.2,
                            batch_size=batch_size,
                            epochs=epochs)
        return history


class MultiClassifier(BinClassifier):
    """Multi-class LSTM classifier
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_number = 0

    def vectorize(self, df: pd.DataFrame) -> Tuple:
        # Vectorize input texts, and automatically infer label number.
        self.label_number = len(df.target.unique())
        cf.info('label_number: {}'.format(self.label_number))
        self.lable_map, y_new = auto_set_label_num(df.target)
        df['target'] = y_new
        return super().vectorize(df)

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Embedding(self.max_feature + 1,
                             self.embedding_dim,
                             input_length=self.max_length),
            LSTM(128, dropout=0.2),
            layers.Dropout(0.4),
            layers.Dense(64),
            layers.Dropout(0.4),
            layers.Dense(64),
            layers.Dropout(0.4),
            layers.Dense(64),
            layers.Dropout(0.4),
            layers.Dense(self.label_number, activation='softmax')
        ])
        print(model.summary())
        model.compile(
            loss='sparse_categorical_crossentropy',
            #   tf.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01),
            metrics=['sparse_categorical_accuracy'])
        # tf.keras.metrics.SparseCategoricalAccuracy()])

        cf.info('model was built.')
        return model


def optimal_batch_size(sequence_size: int) -> int:
    import math
    base = int(math.log(sequence_size + 1, 10))
    batch_size = 1 << (base + 2)
    cf.info('Batch size is set to ', batch_size)
    return batch_size


class NNTouchStone(object):
    def __init__(self,
                 X,
                 y,
                 max_feature: int = 10000,
                 validation_split: float = 0.15,
                 max_length: int = 200,
                 epochs: int = 3,
                 pretrained_vector: str = None):
        """ Text classification with LSTM
        Inputs:
            X: a list of text
            y: labels
            pretrained_vector: pretrained vector file
            max_feature: vocab size of text
            max_length: maximum number of words in one sample
        """
        self.X = X
        self.y = y
        self.Xt = None
        self.yt = None
        self.validation_split = validation_split
        self.max_feature = max_feature
        self.max_length = max_length
        self.epochs = epochs
        self.max_feature = 10000
        self.pretrained_vector = pretrained_vector

    def _tokenize(self):
        self.X, self.Xt, self.y, self.yt, _i, _j = train_test_split(
            self.X,
            self.y,
            np.arange(len(self.X)),
            random_state=63,
            test_size=self.validation_split)

        self.indices = {'train': _i, 'val': _j}

        cf.info('start tokenizing')
        self.X, tokenizer = tokenize(self.X)
        self.Xt = tokenizer.texts_to_sequences(self.Xt)
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index

        if self.max_length < 0:
            self.max_length = int(np.percentile(list(map(len, self.X)), 95))
        cf.info('MAX_LENGTH_SEQUENCE set to {}'.format(self.max_length))
        assert self.max_length >= 2, 'max length is less than 2, check your data.'

        self.X = pad_sequences(self.X, maxlen=self.max_length, padding="pre")
        self.Xt = pad_sequences(self.Xt, maxlen=self.max_length, padding="pre")

        self.input_dim = len(tokenizer.word_index) + 1

    def embed(self, pretrained_vector: str = ''):
        if pretrained_vector:
            self.pretrained_vector = pm.word2vec.load(pretrained_vector)
            cf.info('load {} completes'.format(pretrained_vector))
        values = [
            self.pretrained_vector[word]
            for word in self.pretrained_vector.index_to_key
        ]
        all_embeddings = np.stack(values)
        emb_mean, emb_std = all_embeddings.mean(), all_embeddings.std()
        event = {
            'msg': 'embed {}'.format(pretrained_vector),
            'mean': emb_mean,
            'std': emb_std
        }
        cf.info(event)
        assert self.word_index is not None, 'Tokenize and sequence text first!'

        self.max_feature = len(self.word_index)
        embed_size = all_embeddings.shape[1]
        self.embedding_matrix = np.random.normal(
            emb_mean, emb_std, (self.max_feature, embed_size))

        missed, hit = 0, 0
        for word, i in self.word_index.items():
            if word in self.pretrained_vector:
                embedding_vector = self.pretrained_vector[word]
                self.embedding_matrix[i - 1] = embedding_vector
                hit += 1
            else:
                missed += 1
        cf.info(
            'embed completes, size of embedding matrix {}. Missed {}, hit {}'.
            format(self.embedding_matrix.shape, missed, hit))

        return self.embedding_matrix

    def build_model(self):
        cf.info('Building model...')
        if self.pretrained_vector:
            embedding_matrix = self.embed(self.pretrained_vector)
            ndim = embedding_matrix.shape[1]
            embedding_layer = Embedding(
                len(self.word_index),
                ndim,
                embeddings_initializer=keras.initializers.Constant(
                    embedding_matrix),
                trainable=False,
            )
            int_sequences_input = keras.Input(shape=(None, ), dtype="int64")
            embedded_sequences = embedding_layer(int_sequences_input)
            x = Bidirectional(LSTM(100,
                                   return_sequences=True))(embedded_sequences)
            x = GlobalMaxPool1D()(x)
            x = layers.Dense(100, activation="selu")(x)
            x = layers.Dropout(0.5)(x)
            preds = layers.Dense(1, activation='sigmoid')(x)
            M = keras.Model(int_sequences_input, preds)
        else:
            M = keras.Sequential([
                Embedding(self.input_dim, 100, input_length=self.max_length),
                Bidirectional(LSTM(100)),
                Dropout(0.5),
                Dense(50, activation="selu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid")
            ])

        M.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
        M.summary()
        self.model = M
        return M

    def train(self):
        cf.info('training model')
        weight_path = f'/tmp/best_weights_{cf.uuid()}.h5'
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience=5)
        fit_params = {
            'batch_size': optimal_batch_size(len(self.X)),
            'validation_split': 0.1,
            'epochs': self.epochs,
            'callbacks': [es]
        }
        self.model.fit(self.X, self.y, **fit_params)
        # self.model.load_weights(weight_path)
        self.model.save('/tmp/saved_model.h5')

    def predict(self):
        y_pred = get_binary_prediction(self.model.predict(self.Xt))
        pm.libra.metrics(self.yt, y_pred)
        return self.yt, y_pred

    def benchmark(self):
        self._tokenize()
        self.build_model()
        self.train()
        # return self.predict()
