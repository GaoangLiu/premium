#!/usr/bin/env python
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb#scrollTo=1OXNyH3QzwT5
import argparse
import os

import codefast as cf
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from rich import print
from torch import nn
from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from premium.data.loader import spam_en
from premium.pytorch.data import TextDataset, TextLoader, train_test_val_split
from premium.pytorch.trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Args(object):
    batch_size = 16
    epoches = 3


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        cf.info(df.shape)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                  do_lower_case=True)
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


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertModel()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_id, mask, label):
        return self.model(input_id, mask, label)

    def _update(self, batch, batch_index):
        data, label = batch
        input_id = data['input_ids'].squeeze(1)
        mask = data['attention_mask'].squeeze(1)
        outputs = self.model(input_id, mask, label)
        loss = outputs[0]
        metric = Accuracy().to(self.device)
        acc = metric(outputs[1], label)
        return loss, metric, acc, outputs

    def training_step(self, batch, batch_idx):
        loss, _, acc, _ = self._update(batch, batch_idx)
        self.log('train_acc',
                 acc,
                 prog_bar=True,
                 batch_size=10,
                 on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, acc, _ = self._update(batch, batch_idx)
        self.log('val_acc',
                 acc,
                 prog_bar=True,
                 batch_size=10,
                 on_step=True,
                 on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, acc, _ = self._update(batch, batch_idx)
        self.log('test_acc',
                 acc,
                 prog_bar=True,
                 batch_size=10,
                 on_step=True,
                 on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        data, label = batch
        input_id = data['input_ids'].squeeze(1)
        mask = data['attention_mask'].squeeze(1)
        outputs = self.model(input_id, mask,
                             torch.tensor([0] * len(label)).to(self.device))
        return torch.argmax(outputs[1], dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-08)

    def train_dataloader(self):
        return Args.train

    def val_dataloader(self):
        return Args.val

    def test_dataloader(self):
        return Args.test


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true')
    argparser.add_argument('--test', action='store_true')
    argparser.add_argument('--predict', action='store_true')
    args = argparser.parse_args()

    df = spam_en().train
    X, T, V = train_test_val_split(df, 0.2, 0.5)
    print(T)
    loader = TextLoader(dataset=DataSequence,
                        batch_size=Args.batch_size,
                        shuffle=True,
                        num_workers=8)
    Args.train, Args.val, Args.test = loader(X), loader(
        T, shuffle=False), loader(V, shuffle=False)

    if args.train:
        cf.info("Training")
        checkpoint_callback = ModelCheckpoint(
            dirpath="/data/tmp/",
            save_top_k=2,
            filename='bert_spam_{val_acc:.2f}',
            monitor="val_acc")
            
        progress = RichProgressBar()
        trainer = pl.Trainer(accelerator='gpu',
                             devices=1,
                             max_epochs=Args.epoches,
                             callbacks=[progress, checkpoint_callback])
        trainer.fit(LitModel())
    elif args.test:
        cf.info("Testing")
        model = LitModel.load_from_checkpoint(
            "/data/tmp/bert_spam_val_acc=0.99.ckpt")
        trainer = pl.Trainer(accelerator='gpu', devices=1)
        trainer.test(model)
    elif args.predict:
        cf.info("Predicting")
        trainer = pl.Trainer(accelerator='gpu', devices=1)
        model = LitModel().load_from_checkpoint('/data/tmp/bert_spam_val_acc=0.99.ckpt')
        model.eval()
        preds = trainer.predict(model, Args.test)
        print(preds)
