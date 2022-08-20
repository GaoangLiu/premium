#!/usr/bin/env python
from typing import Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import (AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          BertTokenizerFast, DistilBertModel,
                          DistilBertTokenizer, RobertaTokenizer,
                          TFBertForSequenceClassification, TFBertModel,
                          TFDistilBertForSequenceClassification,
                          TFRobertaModel, TFXLNetModel, XLNetTokenizer)

from premium.models.model_config import KerasCallbacks


class BertDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentences: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentences,
        labels,
        batch_size=32,
        shuffle=True,
        include_targets=True,
        max_length=128,
        bert_model_name="bert-base-cased",
    ):
        self.sentences = sentences
        self.labels = np.array(labels)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.max_length = max_length
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name,
                                                       do_lower_case=True)
        self.indexes = np.arange(len(self.sentences))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentences) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size:(idx + 1) *
                               self.batch_size]
        sentences = self.sentences[indexes]
        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentences.tolist(),
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf",
            truncation=True,
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


class BertClassifier(object):

    def __init__(
        self,
        max_sentence_len: int = 64,
        layer_number: int = 3,
        bert_name: str = "distilbert-base-uncased",
        do_lower_case: bool = True,
        num_labels: int = 2,
        loss: str = None,
        cache_dir: str = "/data/cache",
    ) -> None:
        """ 
        Args:
            max_sentence_len: max length of sentence
            layer_number: number of embedding layers
            bert_name: name of bert model
            do_lower_case: whether to lower case
            num_labels: number of output classes
        """
        self.max_sentence_len = max_sentence_len
        self.bert_name = bert_name
        self.layer_number = layer_number
        self.do_lower_case = do_lower_case
        self.num_labels = num_labels
        self.loss = loss
        self.cache_dir = cache_dir
        if not cf.io.exists(self.cache_dir):
            cf.warning("cache_dir not exists, create one")

    def get_tokenizer(self):
        if self.bert_name in (
                "bert-large-uncased",
                "bert-base-uncased",
                "bert-base-chinese",
                "distilbert-base-uncased",
        ):
            return BertTokenizer.from_pretrained(self.bert_name,
                                                 cache_dir=self.cache_dir)
        elif self.bert_name in ("roberta-large", "roberta-base"):
            return RobertaTokenizer.from_pretrained(self.bert_name,
                                                    cache_dir=self.cache_dir)
        elif self.bert_name in ("xlnet-base-cased", "xlnet-large-cased"):
            return XLNetTokenizer.from_pretrained(self.bert_name,
                                                  cache_dir=self.cache_dir)
        elif self.bert_name in ('bert-base-chinese'):
            return BertTokenizer.from_pretrained(self.bert_name,
                                                 cache_dir=self.cache_dir)
        else:
            raise ValueError("bert_name not supported")

    def get_pretrained_model(self):
        config = BertConfig.from_pretrained(self.bert_name,
                                            output_hidden_states=True,
                                            output_attentions=True)
        if self.bert_name in ("bert-large-uncased", "bert-base-uncased",
                              "distilbert-base-uncased"):
            return TFBertModel.from_pretrained(self.bert_name,
                                               config=config,
                                               cache_dir=self.cache_dir)
        elif self.bert_name in ("bert-base-chinese"):
            return TFBertModel.from_pretrained(self.bert_name,
                                               config=config,
                                               cache_dir=self.cache_dir)
        elif self.bert_name in ("roberta-large", "roberta-base"):
            return TFRobertaModel.from_pretrained(self.bert_name,
                                                  cache_dir=self.cache_dir)
        elif self.bert_name in ("xlnet-base-cased", "xlnet-large-cased"):
            return TFXLNetModel.from_pretrained(self.bert_name,
                                                cache_dir=self.cache_dir)
        else:
            raise ValueError("bert_name not supported")

    def _bert_encode(self, texts: List[str]):
        cf.info("start {} encoding".format(self.bert_name))
        tokenizer = self.get_tokenizer()
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded = tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=self.max_sentence_len,
                                            padding="max_length",
                                            return_attention_mask=True,
                                            truncation=True)
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
        cf.info("{} encoding finished".format(self.bert_name))
        return np.array(input_ids), np.array(attention_masks)

    def _create_model(self, sequence_len: int = 64):
        cf.info("start creating model")
        input_ids = tf.keras.Input(shape=(sequence_len, ),
                                   dtype="int32",
                                   name="input_ids")
        attention_masks = tf.keras.Input(shape=(sequence_len, ),
                                         dtype="int32",
                                         name="attention_masks")

        bert_model = self.get_pretrained_model()
        cf.info("model {} created".format(self.bert_name))
        # Get the final embeddings from the BERT model
        embedding = bert_model([input_ids, attention_masks])
        cf.info("embedding length", len(embedding))
        cf.info("embedding[1] shape", embedding[1].shape)
        embedding = embedding[1]
        embedding = tf.keras.layers.Dense(32, activation='relu')(embedding)
        embedding = tf.keras.layers.Dropout(0.2)(embedding)

        if self.num_labels == 2:     # binary classification
            output = Dense(1, activation="sigmoid")(embedding)
            loss = "binary_crossentropy"
            metrics = ['accuracy']
        else:
            output = Dense(self.num_labels, activation="softmax")(embedding)
            loss = "sparse_categorical_crossentropy"
            metrics = ['sparse_categorical_accuracy']

        if self.loss is not None:
            loss = self.loss
        cf.info("setting output dim to {}".format(self.num_labels))
        cf.info("setting loss to {}".format(loss))

        model = tf.keras.models.Model(inputs=[input_ids, attention_masks],
                                      outputs=output)

        # TODO
        # model.compile(AdamWarmup(decay_steps=decay_steps,
        #                     warmup_steps=warmup_steps, lr=LR),
        #          loss='sparse_categorical_crossentropy',
        #          metrics=['sparse_categorical_accuracy'])
        model.compile(Adam(lr=6e-6), loss=loss, metrics=metrics)
        cf.info("model created")
        return model

    def auto_set_label_num(self, y: List[Union[str,
                                               int]]) -> Tuple[Dict, List[int]]:
        """ Automatically set the number of labels based on the labels.
        If it is binary classification, then new label is like [0, 1, 1, 0], 
        if it is multi-classification, then new label is like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        Return: 
            - A map from old label to new label
            - A list of new label
        """
        label_map = {}
        for yi in y:
            if yi not in label_map:
                label_map[yi] = len(label_map)
        new_y = np.array([label_map[yi] for yi in y])
        self.num_labels = len(label_map)
        return label_map, new_y

    def fit(
        self,
        x,
        y,
        epochs: int = 3,
        batch_size: int = 32,
        early_stop: int = 5,
        validation_split: float = 0.3,
    ) -> Tuple[tf.keras.Model, Dict]:
        """
        Args:
            x: list of str
            y: list of int
            batch_size(int): batch size, if set -1, will try and found the max batch 
            size that suits the gpu.
        """
        label_map, y = self.auto_set_label_num(y)
        ids, masks = self._bert_encode(x)
        msg = {
            'label_number': self.num_labels,
            'label_map':
            label_map,
            'input_ids_shape':
            len(ids) if isinstance(ids, list) else ids.shape,
            'input_masks_shape':
            len(masks) if isinstance(masks, list) else masks.shape
        }
        cf.info(msg)

        model = self._create_model(sequence_len=self.max_sentence_len)
        cf.info("model summary:")
        print(model.summary())

        # trained the classfier on use layer
        # filepath = "models/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stop,
            restore_best_weights=True,
            # restore model weights from the epoch with the best value of the monitored quantity
            verbose=1,
        )

        mcp_save = ModelCheckpoint(filepath="/tmp/",
                                   save_weights_only=True,
                                   monitor="val_loss",
                                   mode="auto")

        reduce_lr_loss = ReduceLROnPlateau(monitor="val_loss",
                                           factor=0.1,
                                           patience=5,
                                           verbose=1,
                                           mode="min")

        csv_logger = CSVLogger("/tmp/keraslog.csv", append=True, separator=";")
        history = model.fit(
            [ids, masks],
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, mcp_save, reduce_lr_loss, csv_logger],
        )
        self.model = model
        return history

    def predict(self, xt):
        cf.info("start predicting")
        tids, tmasks = self._bert_encode(xt)
        preds = self.model.predict([tids, tmasks])
        preds = np.round(preds).astype(int)
        cf.info("predict finished")
        return preds

    def show_history(self, history):
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()


def bert_benchmark(df: pd.DataFrame,
                   epochs: int = 1,
                   bert_name: str = 'bert-base-chinese'):
    """ A Bert classifier wrapper for faster benchmark. 
    """
    assert 'text' in df.columns, 'text column not found'
    assert 'target' in df.columns, 'target column not found'
    bc = BertClassifier(bert_name=bert_name)
    history = bc.fit(df['text'], df['target'], epochs=epochs)
    return history


def map_sample_to_dict(input_ids, attention_masks, token_type_ids, label):
    return (
        {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks,
        },
        label,
    )


class Dataset(object):

    def __init__(self, csv_file: str, ratio: float = -1):
        self.df = pd.read_csv(csv_file)
        if ratio > 0:
            self.df = self.df.sample(frac=ratio)
        cf.info("data loaded: {}".format(csv_file))
        assert "label" in self.df.columns, "rename your dataset to have label column"

    def split(self, val_split: float, test_split: float = 0):
        X, Xv = train_test_split(
            self.df,
            test_size=val_split + test_split,
            stratify=self.df["label"],
            random_state=42,
        )
        if test_split == 0:
            return X, Xv, None
        r = test_split / (val_split + test_split)
        Xv, Xt = train_test_split(Xv,
                                  test_size=r,
                                  stratify=Xv["label"],
                                  random_state=43)
        cf.info("Data splited: train: {}, val: {}, test: {}".format(
            len(X), len(Xv), len(Xt)))
        return X, Xv, Xt