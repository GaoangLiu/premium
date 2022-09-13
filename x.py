#!/usr/bin/env python
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)

imdb = load_dataset('imdb')
print(imdb['train'][0])
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
train_dataset = train_dataset.map(tokenize,
                                  batched=True,
                                  batch_size=len(train_dataset))
test_dataset = test_dataset.map(tokenize,
                                batched=True,
                                batch_size=len(train_dataset))
train_dataset.set_format('torch',
                         columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch',
                        columns=['input_ids', 'attention_mask', 'label'])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels,
                                                               preds,
                                                               average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset)

trainer.train()

trainer.evaluate()
