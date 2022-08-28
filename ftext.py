#!/usr/bin/env python
import codefast as cf
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow import keras

df = pd.read_csv('localdata/tmp.csv')
df = df.sample(frac=0.5)

max_feature = 20000
max_length = 100
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_feature,
    output_mode='int',
    output_sequence_length=max_length)
vectorize_layer.adapt(df.text)
xs = vectorize_layer(['are you okay'])
print(xs)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df.text)
text_sequences = tokenizer.texts_to_sequences(df.text)
text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences,
                                                               padding='post')
targets = set(df.target.values)
target2idx = {t: i for i, t in enumerate(targets)}
idx2target = {i: t for i, t in enumerate(targets)}

vocab_size = len(vectorize_layer.get_vocabulary())
label_number = len(targets)
event = {
    'vocab_size': vocab_size,
    'label_number': label_number,
    'max_length': max_length,
    'target2idx': target2idx,
    'idx2target': idx2target
}
cf.info(event)

df['target'] = df.target.map(target2idx)

file_name = 'glove.twitter.27B.25d.txt'
file_name = 'glove.6B.100d.txt'
fasttext = KeyedVectors.load_word2vec_format(file_name)
embedding_dim = 100
print('load complete')

# X = vectorize_layer(df.text.values)
X = text_sequences
y = df.target
y = tf.keras.utils.to_categorical(y, num_classes=label_number)
max_length = X.shape[1]

vocab_size = len(tokenizer.word_index) + 1
weight_matrix = np.zeros((vocab_size, embedding_dim))

# for i, word in enumerate(vectorize_layer.get_vocabulary()):
for word, i in tokenizer.word_index.items():
    try:
        weight_matrix[i] = fasttext[word]
    except KeyError:
        weight_matrix[i] = np.random.uniform(-5, 5, embedding_dim)

sentence_input = tf.keras.layers.Input(shape=(max_length, ))
x = tf.keras.layers.Embedding(
    vocab_size,
    embedding_dim,
     #   weights=[weight_matrix],
    input_length=max_length,
)(sentence_input)

x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.LSTM(100)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(100, activation='selu')(x)
output = tf.keras.layers.Dense(label_number, activation='softmax')(x)
model = tf.keras.Model(sentence_input, output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, validation_split=0.2, epochs=10, batch_size=32)
model.save('localdata/model.h5')
