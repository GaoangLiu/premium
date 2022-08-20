#!/usr/bin/env python
from typing import List, Tuple

import codefast as cf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Dense, Embedding, Lambda,
                                     Reshape, add)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.utils import to_categorical

import premium as pm
from premium.corpus import bookreader


# build vocabulary of unique words
class SimpleCBOW(object):
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
        cnt = 1
        for wl in self.words_list:
            lst = []
            for w in wl:
                if w not in self.word2id:
                    self.word2id[w] = cnt
                    self.id2word[cnt] = w
                    cnt += 1
                lst.append(self.word2id[w])
            self.word_ids.append(lst)
        self.vocab_size = len(self.word2id)
        cf.info("words list size: ", len(self.words_list))
        cf.info("Vocabulary size: ", self.vocab_size)

    def generate_context_word_pairs(self, corpus: List[List[int]]):
        """
        corpus: list of list of word id(int)
        """
        context_length = self.window_size * 2
        for words in corpus:
            sentence_length = len(words)
            for index, word in enumerate(words):
                context_words = []
                label_word = []
                start = index - self.window_size
                end = index + self.window_size + 1

                context_words.append([
                    words[i] for i in range(start, end)
                    if 0 <= i < sentence_length and i != index
                ])
                label_word.append(word)
                x = sequence.pad_sequences(context_words,
                                           maxlen=context_length)
                y = to_categorical(label_word, self.vocab_size)
                yield (x, y)

    def build_model(self):
        cbow = Sequential()
        cbow.add(
            Embedding(input_dim=self.vocab_size,
                      output_dim=self.embed_size,
                      input_length=self.window_size * 2))
        cbow.add(
            Lambda(lambda x: K.mean(x, axis=1),
                   output_shape=(self.embed_size, )))
        cbow.add(Dense(self.vocab_size, activation="softmax"))
        cbow.compile(loss="categorical_crossentropy", optimizer="rmsprop")
        # view model summary
        cf.info('cbow model summary:')
        print(cbow.summary())
        return cbow

    def train(self) -> Tuple[List, Sequential]:
        cbow = self.build_model()
        for epoch in range(self.epoches):
            loss = 0.0
            for x, y in self.generate_context_word_pairs(corpus=self.word_ids):
                loss += cbow.train_on_batch(x, y)
            cf.info("epoch {} loss {}".format(epoch, loss))
        cf.info('weights length:', len(cbow.get_weights()))
        weights = cbow.get_weights()[0]
        weights = weights[1:]
        print(weights.shape)
        print(
            pd.DataFrame(weights,
                         index=list(self.id2word.values())[1:]).head())
        return weights, cbow

    def predict(self):
        pass


class SkipGrams(SimpleCBOW):
    def __init__(self,
                 words_list: List[List[str]],
                 embed_size: int = 100,
                 window_size: int = 2,
                 epoches: int = 30) -> None:
        super().__init__(words_list, embed_size, window_size, epoches)

    def build_model(self):
        # build skip-gram architecture
        word_model = Sequential()
        word_model.add(
            Embedding(self.vocab_size,
                      self.embed_size,
                      embeddings_initializer="glorot_uniform",
                      input_length=1))
        word_model.add(Reshape((self.embed_size, )))

        context_model = Sequential()
        context_model.add(
            Embedding(self.vocab_size,
                      self.embed_size,
                      embeddings_initializer="glorot_uniform",
                      input_length=1))
        context_model.add(Reshape((self.embed_size, )))

        merged_output = add([word_model.output, context_model.output])

        model_combined = Sequential()
        model_combined.add(
            Dense(1, kernel_initializer="glorot_uniform",
                  activation="sigmoid"))

        final_model = Model([word_model.input, context_model.input],
                            model_combined(merged_output))
        final_model.compile(loss="mean_squared_error", optimizer="rmsprop")
        final_model.summary()
        return final_model

    def tokenize(self):
        super().tokenize()
        self.skip_grams = [
            skipgrams(wid,
                      vocabulary_size=self.vocab_size,
                      window_size=self.window_size) for wid in self.word_ids
        ]

    def train(self):
        model = self.build_model()
        for epoch in range(self.epoches):
            loss = 0
            for i, elem in enumerate(self.skip_grams):
                if not elem[0]: continue
                pair_first_elem = np.array(list(zip(*elem[0]))[0],
                                           dtype='int32')
                pair_second_elem = np.array(list(zip(*elem[0]))[1],
                                            dtype='int32')
                labels = np.array(elem[1], dtype='int32')
                X = [pair_first_elem, pair_second_elem]
                Y = labels
                if i % 10000 == 0:
                    cf.info(
                        'Processed {} (skip_first, skip_second, relevance) pairs'
                        .format(i))
                loss += model.train_on_batch(X, Y)
            cf.info('Epoch:', epoch, 'Loss:', loss)


