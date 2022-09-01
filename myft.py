#!/usr/bin/env python
from typing import List, Tuple

import codefast as cf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Embedding, Lambda,
                                     Reshape, add)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.utils import to_categorical

import premium as pm
from premium.corpus import bookreader


# build vocabulary of unique words
class myFastText(object):

    def __init__(self,
                 words_list: List[List[str]],
                 embed_size: int = 100,
                 window_size: int = 2,
                 epoches: int = 30) -> None:
        self.words_list = words_list
        self.embed_size = embed_size
        self.window_size = window_size
        self.word2id = {"PAD": 0}
        self.id2word = {0: "PAD"}
        self.word_ids = []
        self.vocab_size = 0
        self.epoches = epoches

    def tokenize(self):
        self.vocab_size = 1
        for wl in self.words_list:
            lst = []
            for w in wl:
                if w not in self.word2id:
                    self.word2id[w] = self.vocab_size
                    self.id2word[self.vocab_size] = w
                    self.vocab_size += 1
                lst.append(self.word2id[w])
            self.word_ids.append(lst)
        cf.info("words list size: ", len(self.words_list))
        cf.info("Vocabulary size: ", self.vocab_size)

    def generate_context_word_pairs(self, word_id_list: List[List[int]]):
        """
        word_id_list: list of list of word id(int)
        """
        context_length = self.window_size * 2
        for word_ids in word_id_list:
            sentence_length = len(word_ids)
            for index, word in enumerate(word_ids):
                context_words = []
                label_word = []
                start = index - self.window_size
                end = index + self.window_size + 1

                context_words.append([
                    word_ids[i] for i in range(start, end)
                    if 0 <= i < sentence_length and i != index
                ])
                label_word.append(word)
                x = sequence.pad_sequences(context_words,
                                           maxlen=context_length)
                y = to_categorical(label_word, self.vocab_size)
                yield (x, y)

    def build_model(self, n_features: int = 100000, embedding_dimensions: int = 300, input_length: int = 20):
        model = Sequential()
        model.add(Embedding(n_features,
                            embedding_dimensions,
                            input_length=input_length))

        model.add(keras.layers.GlobalAveragePooling1D())

        model.add(Dense(1, activation='sigmoid'))
        cf.info('model summary:')
        print(model.summary())
        return model

    def train(self) -> Tuple[List, Sequential]:
        model = self.build_model()
        for epoch in range(self.epoches):
            loss = 0.0
            for x, y in self.generate_context_word_pairs(self.word_ids):
                loss += model.train_on_batch(x, y)
            cf.info("epoch {} loss {}".format(epoch, loss))
        cf.info('weights length:', len(model.get_weights()))
        weights = model.get_weights()[0]
        weights = weights[1:]
        print('weights shape:', weights.shape)
        print(
            pd.DataFrame(weights,
                         index=list(self.id2word.values())[1:]).head())
        return weights, model

    def predict(self):
        pass
