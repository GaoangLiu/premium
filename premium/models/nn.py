#!/usr/bin/env python
import codefast as cf
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout,
                                     Embedding, GlobalMaxPool1D, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import premium as pm
from premium.data.postprocess import get_binary_prediction
from premium.data.preprocess import pad_sequences, tokenize
from premium.models.model_config import KerasCallbacks

tf.compat.v1.disable_eager_execution()


class BiLSTM(object):

    def __init__(self,
                 max_feature: int = 50000,
                 max_len: int = 300,
                 embed_size: int = 200,
                 X=None,
                 y=None) -> None:
        self.max_feature = max_feature
        self.max_length = max_len
        self.embed_size = embed_size
        self.X = X
        self.y = y
        self.sequence = None
        self.tokenizer = None
        self.word_index = None
        self.keras_callbacks = KerasCallbacks()

    def tokenize_sequence(self):
        self.X, tokenizer = tokenize(self.X, max_feature=self.max_feature)
        self.sequence = sequence.pad_sequences(self.X, maxlen=self.max_length)
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index
        cf.info('{} of unique tokens were found'.format(len(self.word_index)))
        return self.tokenizer

    def embed(self, pretrained_vector: str = 'glove-wiki-gigaword-50'):
        self.pretrained_vector = pm.word2vec.load(pretrained_vector)
        cf.info('load {} completes'.format(pretrained_vector))

        all_embeddings = np.stack(list(self.pretrained_vector.values()))
        emb_mean, emb_std = all_embeddings.mean(), all_embeddings.std()
        assert self.word_index is not None, 'Tokenize and sequence text first!'

        self.max_feature = min(self.max_feature, len(self.word_index))
        self.embedding_matrix = np.random.normal(
            emb_mean, emb_std, (self.max_feature, self.embed_size))

        hit, missed = 0, 0
        for word, i in self.word_index.items():
            if i < self.max_feature:
                embedding_vector = self.pretrained_vector.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector
                    hit += 1
                else:
                    missed += 1
        cf.info(
            'embed completes, size of embedding matrix {}. Missed {}, hit {}'.
            format(self.embedding_matrix.shape, missed, hit))

        return self.embedding_matrix

    def build_model(self):
        inp = Input(shape=(self.max_length, ))
        x = Embedding(self.max_feature,
                      self.embed_size,
                      weights=[self.embedding_matrix])(inp)
        # x = Embedding(max_feature, embed_size)(inp)
        x = Bidirectional(LSTM(100, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.5)(x)
        opt = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=opt)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        self.model = model
        cf.info('model was built.')
        return model

    def fit(self, batch_size: int = 32, epochs: int = 5):
        callbacks = self.keras_callbacks.some(
            ['early_stopping', 'reduce_lr', 'csv_logger'])
        self.model.fit(self.sequence,
                       self.y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.15,
                       callbacks=callbacks,
                       use_multiprocessing=True)

        return self.model

    def predict(self, X_te):
        return self.model.predict(X_te,
                                  batch_size=1024,
                                  verbose=1,
                                  use_multiprocessing=True)


def build_bilstm(maxlen: int, max_feature: int, embed_size: int):
    inp = Input(shape=(maxlen, ))
    layer = Embedding(max_feature, embed_size)(inp)
    layer = Bidirectional(LSTM(50, return_sequences=True))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(50, activation="relu")(layer)
    layer = Dropout(0.1)(layer)
    opt = Dense(6, activation="sigmoid")(layer)
    model = Model(inputs=inp, outputs=opt)
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

    def __init__(self, X, y, max_feature: int = 10000, validation_split: float = 0.15, max_length: int = -1, epochs: int = 3):
        """ Text classification with LSTM
        Inputs: 
            X: a list of text
            y: labels
        """
        self.X = X
        self.y = y
        self.Xt = None
        self.yt = None
        self.validation_split = validation_split
        self.max_length = max_length
        self.epochs = epochs
        self.max_feature = 10000

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
        values = [self.pretrained_vector[word]
                  for word in self.pretrained_vector.index_to_key]
        all_embeddings = np.stack(values)
        emb_mean, emb_std = all_embeddings.mean(), all_embeddings.std()
        event = {'msg': 'embed {}'.format(
            pretrained_vector), 'mean': emb_mean, 'std': emb_std}
        assert self.word_index is not None, 'Tokenize and sequence text first!'

        self.max_feature = len(self.word_index)
        embed_size = all_embeddings.shape[1]
        self.embedding_matrix = np.random.normal(
            emb_mean, emb_std, (self.max_feature, embed_size))

        missed, hit = 0, 0
        for word, i in self.word_index.items():
            if word in self.pretrained_vector:
                embedding_vector = self.pretrained_vector[word]
                self.embedding_matrix[i-1] = embedding_vector
                hit += 1
            else:
                missed += 1
        cf.info(
            'embed completes, size of embedding matrix {}. Missed {}, hit {}'.
            format(self.embedding_matrix.shape, missed, hit))

        return self.embedding_matrix

    def build_model(self):
        cf.info('Building model...')
        from tensorflow.keras import layers
        from tensorflow import keras
        embedding_matrix = self.embed('glove-twitter-25')
        embedding_layer = Embedding(
            len(self.word_index),
            25,
            embeddings_initializer=keras.initializers.Constant(
                embedding_matrix),
            trainable=False,
        )
        int_sequences_input = keras.Input(shape=(None,), dtype="int64")
        embedded_sequences = embedding_layer(int_sequences_input)
        x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        preds = layers.Dense(1, activation='sigmoid')(x)
        M = keras.Model(int_sequences_input, preds)

        # M = Sequential([
        #     keras.Input(shape=(None,), dtype="int64"),
        #     embedding_layer(),
        #     # Embedding(self.input_dim, 100, input_length=self.max_length),
        #     Bidirectional(LSTM(100)),
        #     Dropout(0.5),
        #     Dense(50, activation="relu"),
        #     Dropout(0.5),
        #     Dense(1, activation="sigmoid")
        # ])

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
        checkpoint = ModelCheckpoint(weight_path,
                                     monitor='val_loss',
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=True,
                                     save_freq=1)
        fit_params = {
            'batch_size': optimal_batch_size(len(self.X)),
            'validation_split': 0.1,
            'epochs': self.epochs,
            'callbacks': [es]
        }
        self.model.fit(self.X, self.y, **fit_params)
        self.model.load_weights(weight_path)
        self.model.save('/tmp/saved_model.h5')

    def predict(self):
        y_pred = get_binary_prediction(self.model.predict(self.Xt))
        pm.libra.metrics(self.yt, y_pred)
        return self.yt, y_pred

    def pipeline(self):
        self._tokenize()
        self.build_model()
        self.train()
        # return self.predict()
