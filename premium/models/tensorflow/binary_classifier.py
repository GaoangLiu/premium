import tensorflow as tf
import codefast as cf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
import numpy as np


class BinaryClassifier(object):
    def __init__(self,
                 feature_number: int,
                 epoches: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 test_split: float = 0.2,
                 model_kernel_initializer='glorot_uniform',
                 model_activation='relu'):
        self.feature_number = feature_number
        self.epoches = epoches
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.model_kernel_initializer = model_kernel_initializer
        self.model_activation = model_activation

    def create_model(self):
        _input = keras.Input(shape=(self.feature_number, ), dtype='float32')
        v = _input

        v = keras.layers.Dense(128, activation=self.model_activation)(_input)
        v = keras.layers.Dropout(0.4)(v)
        v = keras.layers.BatchNormalization()(v)
        v = keras.layers.Dense(32, activation=self.model_activation)(v)
        v = keras.layers.Dropout(0.4)(v)
        output = keras.layers.Dense(1, activation="sigmoid")(v)
        model = keras.Model(_input, output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, x, y):
        self.model = self.create_model()
        early_stopping = EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       patience=10,
                                       restore_best_weights=True,
                                       verbose=1)

        mcp_save = ModelCheckpoint(filepath='/tmp/',
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   mode='auto')

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=5,
                                           verbose=1,
                                           mode='min')
        self.model.fit(x,
                       y,
                       epochs=self.epoches,
                       batch_size=self.batch_size,
                       validation_split=self.validation_split,
                       callbacks=[early_stopping, reduce_lr_loss, mcp_save])
        return self.model

    def predict(self, xt):
        return (self.model.predict(xt) > 0.5).astype(np.int)