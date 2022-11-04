#!/usr/bin/env python
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

# https://bhuvana-kundumani.medium.com/implementation-of-simcse-for-unsupervised-approach-in-pytorch-a3f8da756839


class wikiDataset(Dataset):
    def __init__(self, csv_path, training=True, full=False):
        dataset_df = pd.read_csv(csv_path, names=["text"])
        dataset_df.dropna(inplace=True)
        source_texts = dataset_df["text"].values
        target_texts = dataset_df["text"].values
        data = list(zip(source_texts, target_texts))
        if full:
            self.data = data
        else:
            train_data, val_data = train_test_split(data,
                                                    test_size=0.15,
                                                    random_state=42,
                                                    shuffle=False)
            self.data = train_data if training else val_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def process_batch(txt_list, tokenizer, max_len=200):
    source_ls = [source for source, target in txt_list]
    target_ls = [target for source, target in txt_list]
    source_tokens = tokenizer(source_ls,
                              truncation=True,
                              padding="max_length",
                              max_length=max_len)
    target_tokens = tokenizer(target_ls,
                              truncation=True,
                              padding="max_length",
                              max_length=max_len)
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for i in range(len(source_tokens["input_ids"])):
        input_ids.append(source_tokens["input_ids"][i])
        input_ids.append(target_tokens["input_ids"][i])
        attention_mask.append(source_tokens["attention_mask"][i])
        attention_mask.append(target_tokens["attention_mask"][i])
        token_type_ids.append(source_tokens["token_type_ids"][i])
        token_type_ids.append(target_tokens["token_type_ids"][i])
    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(
        token_type_ids)


data = wikiDataset('/tmp/wiki1m_for_simcse.txt', training=True, full=True)
print(len(data))
