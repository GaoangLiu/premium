#!/usr/bin/env python
import json
import os
import random
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from functools import reduce
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import joblib
import nltk
import numpy as np
import pandas as pd
from rich import print
from torchtext import data
from torchtext.vocab import vocab
from torchtext.vocab.vectors import GloVe, Vectors

sentence = "Natural language processing strives to build machines that understand \
    and respond to text or voice data and respond with text or speech of their own in much the same way humans do."

sentences_list = sentence.split(" ")     # 切分句子

counter = Counter(sentences_list)     # 统计计数
sorted_by_freq_tuples = sorted(counter.items(),
                               key=lambda x: x[1],
                               reverse=True)     # 构造成可接受的格式：[(单词,num), ...]
ordered_dict = OrderedDict(sorted_by_freq_tuples)
print(ordered_dict)
# 开始构造 vocab
my_vocab = vocab(ordered_dict, specials=["<UNK>", "<SEP>"])
my_vocab.set_default_index(-1)

print(my_vocab, my_vocab['to'], my_vocab['and'])


def get_tok_ids(sentence: str, vocab: vocab) -> List[int]:
    tokens = sentence.split(' ')
    tok_ids = [vocab[tok] for tok in tokens]
    return tok_ids


sentence = "Natural language processing strives to build machines that understand next next"
tok_ids = get_tok_ids(sentence, my_vocab)
print(tok_ids)

# examples = ['chip', 'token']
# examples = ['飞书', '微信']
# vec = Vectors(name='/data/corpus/tencent100.txt', cache='/data/.vector_data/')
# ret = vec.get_vecs_by_tokens(examples, lower_case_backup=True)
# print(ret)

# def get_vecs_by_sentence(sentence: str, vec: Vectors) -> np.ndarray:
#     tokens = sentence.split(' ')
#     vecs = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
#     return vecs
