#!/usr/bin/env python
from typing import List

import codefast as cf
import numpy as np
import pandas as pd
import tensorflow as tf


class TextNN(object):

    def __init__(self,
                 vocab_size: int = 20000,
                 embedding_size: int = 100,
                 max_length: int = 200) -> None:
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length = max_length


class TextCNN(TextNN):

    def __init__(self,
                 vocab_size: int = 20000,
                 embedding_size: int = 100,
                 max_length: int = 200) -> None:
        super().__init__(vocab_size, embedding_size, max_length)

    def __call__(self, *args, **kwargs):
        pass

    def vectorize(self, texts: List[str],
                  vocab_size: int) -> tf.keras.layers.TextVectorization:
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=100,
        )
        text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(128)
        vectorizer.adapt(text_ds)
        return vectorizer

    def build_model(
        self,
        vocab_size: int = 10000,
        embedding_size: int = 100,
        vectorizer: tf.keras.layers.TextVectorization = None,
    ):
        model = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(vocab_size, embedding_size),
            tf.keras.layers.Conv1D(embedding_size, 2, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy'],
        )
        return model

    def baseline(self,
                 df: pd.DataFrame,
                 test_size: float = 0.15,
                 epochs: int = 10,
                 batch_size: int = 32) -> None:
        """
        Inputs:
            df: data to be tested
            test_size: validation ratio 
        """
        X = df.sample(frac=1 - test_size, random_state=200)
        Xt = df.drop(X.index)
        X, y = X['text'].values, X['target'].values
        Xt, yt = Xt['text'].values, Xt['target'].values
        vectorizer = self.vectorize(X, self.vocab_size)
        model = self.build_model(vectorizer=vectorizer,
                                 embedding_size=self.embedding_size,
                                 vocab_size=self.vocab_size)
        model.summary()
        history = model.fit(X,
                            y,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(Xt, yt))
        return history


if __name__ == "__main__":
    # downloader.imdb()
    import pandas as pd

    from premium.data.datasets import downloader
    from premium.models.nn import BinClassifier
    df = pd.read_csv('/tmp/imdb_sentiment.csv')

    cli = TextCNN(vocab_size=30000, embedding_size=100, max_length=300)
    cli = BinClassifier()
    cli.baseline(df)
    # cli.touchstone(df)
