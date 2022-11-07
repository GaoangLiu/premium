#!/usr/bin/env python
#coding=utf8
from premium.data import loader
import json
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import joblib
import numpy as np
import pandas as pd
import torch
from descriptors import cachedclassproperty, cachedproperty, classproperty
from rich import print
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizerFast
import premium as pm
# refer: https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model_name = 'bert-base-chinese'
# tokenizer = BertTokenizerFast.from_pretrained(model_name)
# texts = ["其实问题很好解决，就是花更多的时间去学习。","嗯现在我没有像大学刚毕业那样，满脑子的冲劲，得过且过的一天"]
# texts_tok = tokenizer(texts, padding=True, truncation=True, max_length=50, return_tensors='pt')
# print(texts_tok)
# print(texts_tok['input_ids'].shape)

# print(texts_tok.word_ids())


Cfg = pm.load_yaml('data/config/ner_cn.yaml')
print(Cfg)
print(Cfg.defaults.ner_label)




