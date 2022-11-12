#!/usr/bin/env python
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb#scrollTo=1OXNyH3QzwT5
from typing import Callable

import codefast as cf
import pandas as pd
import torch
from rich import print
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from premium.data.loader import imdb_sentiment, spam_en
from premium.pytorch import BatchCollector
from premium.pytorch.data import TextDataset, train_test_val_split
from premium.pytorch.tokenizer import VocabTokenizer
from premium.pytorch.trainer import Trainer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Configs(object):
    MAX_WORDS = 10000
    MAX_LEN = 100
    BATCH_SIZE = 32
    EMB_SIZE = 128
    HID_SIZE = 128
    DROPOUT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoches = 2


from transformers import AutoModelForSequenceClassification, AutoTokenizer

df = spam_en().train
X, T, V = train_test_val_split(df, 0.2, 0.5)
print(T)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                          do_lower_case=True)


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        cf.info(df.shape)

        self.texts = [
            tokenizer(t,
                      padding='max_length',
                      max_length=120,
                      truncation=True,
                      return_tensors="pt") for t in df.text
        ]
        self.labels = df.label.tolist()

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.labels[idx]

        return batch_data, batch_labels


class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2)

    def forward(self, input_id, mask, label):
        return self.bert(input_ids=input_id,
                         attention_mask=mask,
                         labels=label,
                         return_dict=False)


train = DataLoader(DataSequence(X), batch_size=Configs.BATCH_SIZE, shuffle=True, num_workers=8)
val = DataLoader(DataSequence(V), batch_size=1, shuffle=False)
test = DataLoader(DataSequence(T), batch_size=1, shuffle=False)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
model = BertModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-08)
from tqdm import trange, tqdm
from torchmetrics import Accuracy
import pytorch_lightning as pl 

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertModel()
        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, input_id, mask, label):
        return self.model(input_id, mask, label)

    def training_step(self, batch, batch_idx):
        data, label = batch
        input_id = data['input_ids'].squeeze(1)
        mask = data['attention_mask'].squeeze(1)
        outputs = self.model(input_id, mask, label)
        loss = outputs[0]
        self.metric(outputs[1], label)
        self.log("performance", {"acc": self.metric, "loss": loss}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        input_id = data['input_ids'].squeeze(1)
        mask = data['attention_mask'].squeeze(1)
        outputs = self.model(input_id, mask, label)
        loss = outputs[0]
        preds = outputs[1]
        accuracy = Accuracy()
        acc = accuracy(preds, label)
        self.log('accuracy', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_id, mask, label = batch
        outputs = self.model(input_id, mask, label)
        loss = outputs[0]
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-08)

    def train_dataloader(self):
        return train

    def val_dataloader(self):
        return val

    def test_dataloader(self):
        return test


trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=2)
trainer.fit(LitModel())
exit(0)

for e in range(Configs.epoches):
    model.train()
    accuracy = Accuracy().cuda()
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    device = Configs.DEVICE

    for data, label in tqdm(train):
        label = label.to(device)
        input_id = data['input_ids'].squeeze(1).to(device)
        mask = data['attention_mask'].squeeze(1).to(device)
        loss, logits = model(input_id, mask, label)
        preds = torch.argmax(logits, dim=1)
        accuracy(preds, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tr_loss += loss.item()
        nb_tr_examples += input_id.size(0)
        nb_tr_steps += 1
    cf.info({
        "epoch": e, 
        "train_loss": tr_loss/nb_tr_steps, 
        "train_acc": accuracy.compute()
    })

    model.eval()
    accuracy.reset()
    for data, label in tqdm(test):
        label = label.to(device)
        input_id = data['input_ids'].squeeze(1).to(device)
        mask = data['attention_mask'].squeeze(1).to(device)
        loss, logits = model(input_id, mask, label)
        preds = torch.argmax(logits, dim=1)
        accuracy(preds, label)
    cf.info({
        "epoch": e, 
        "test_loss": tr_loss/nb_tr_steps, 
        "test_acc": accuracy.compute()
    })


