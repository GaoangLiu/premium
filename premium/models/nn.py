#!/usr/bin/env python
import codefast as cf
import numpy as np  # linear algebra
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout,
                                     Embedding, GlobalMaxPool1D, Input)
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import premium as pm
from premium.data.postprocess import get_binary_prediction
from premium.data.preprocess import pad_sequences, tokenize
from premium.models.model_config import KerasCallbacks
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union


def to_chars(text: str):
    """For splitting the input text, expecially non-English text. 
    refer https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
    for more detail
    """
    return tf.strings.unicode_split(text,
                                    input_encoding='UTF-8',
                                    errors="ignore")


class BiLSTM(object):
    def __init__(self,
                 max_feature: int = 50000,
                 max_length: int = 300,
                 embedding_dim: int = 200,
                 vectorizer_split_strategy: Union[Callable, str] = 'whitespace') -> None:
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
        args = {'max_length': self.max_length,
                'embeding_dim': self.embedding_dim, 'max_feature': self.max_feature}
        cf.info('args: {}'.format(args))

    def vectorize(self, df: pd.DataFrame) -> Tuple:
        """Split dataset into (train, test). Vectorize on train corpus.
        """
        df_train, df_test = train_test_split(df)
        train_dataset = tf.data.Dataset.from_tensor_slices(df_train.text)

        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=self.max_feature,
            output_mode='int',
            output_sequence_length=self.max_length, split=self.vectorizer_split_strategy)
        vectorize_layer.adapt(train_dataset.batch(64))

        cf.info('vectorizing complete')
        return (vectorize_layer(df_train.text),
                df_train.target), (vectorize_layer(df_test.text),
                                   df_test.target)

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Embedding(self.max_feature+1, self.embedding_dim),
            layers.SpatialDropout1D(0.2),
            layers.Bidirectional(layers.LSTM(
                128, return_sequences=True, dropout=0.2)),
            layers.Dropout(0.4),
            layers.Dense(64),
            layers.Dropout(0.4),
            layers.Dense(1)
        ])
        print(model.summary())
        model.compile(loss=tf.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=['accuracy'])

        cf.info('model was built.')
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

    def benchmark(self, df: pd.DataFrame, batch_size: int = 32, epochs: int = 3):
        """Do a quick benchmark on given dataset in format
        target, text
        1, some text
        """
        assert 'target' in df.columns, 'target must be in columns'
        assert 'text' in df.columns, 'text must be in columns'
        train_ds, test_ds = self.vectorize(df)
        model = self.build_model()
        history, _ = self.fit(train_ds, test_ds, model,
                              epochs=epochs, batch_size=batch_size)
        return history


def build_bilstm(maxlen: int, max_feature: int, embed_size: int):
    inp = Input(shape=(maxlen, ))
    layer = Embedding(max_feature, embed_size)(inp)
    layer = Bidirectional(LSTM(50, return_sequences=True))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(50, activation="relu")(layer)
    layer = Dropout(0.1)(layer)
    opt = Dense(6, activation="sigmoid")(layer)
    model = keras.Model(inputs=inp, outputs=opt)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

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
