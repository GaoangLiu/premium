#!/usr/bin/env python
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import torch

from tqdm import tqdm
# import wandb
# wandb.init(project='trainer')

class TrainerInterface(ABC):
    @abstractmethod
    def evaluate(self, valid_dataloader):
        pass

    @abstractmethod
    def train(self, train_dataloader, epoch):
        pass


class Metrics(object):
    def __init__(self, epoch: int):
        self.loss = 0
        self.n_correct = 0
        self.n_total = 0
        self.cur_loss = 0
        self.cur_acc = 0
        self.epoch = epoch

    def update(self, loss, n_correct, n_total):
        self.loss += loss
        self.n_correct += n_correct
        self.n_total += n_total
        self.calc()

    def calc(self):
        self.cur_loss = self.loss / self.n_total
        self.cur_acc = self.n_correct / self.n_total
        return self.cur_loss, self.cur_acc

    def __str__(self):
        return f'| epoch {self.epoch:3d} | loss {self.cur_loss:5.2f} | acc {self.cur_acc:5.2f}'


class Trainer(TrainerInterface):
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_dataloader, epoch):
        self.model.to(self.device)
        self.model.train()

        for e in range(epoch):
            metric = Metrics(e)
            for batch in tqdm(train_dataloader):
                text, offsets, label = batch['text'], batch['offset'], batch['label']
                self.optimizer.zero_grad()
                output = self.model(text, offsets)
                loss = self.criterion(output, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                metric.update(loss.item(), (output.argmax(1) == label).sum().item(), label.size(0))

            cf.info(metric)

    def evaluate(self, valid_dataloader):
        pass

