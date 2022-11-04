#!/usr/bin/env python
import os
import random
import json
import re
import sys
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import numpy as np
import pandas as pd
from rich import print
from typing import List, Union, Callable, Set, Dict, Tuple, Optional
from premium.models.bert import BertClassifier
from premium.data.datasets import downloader
# downloader.get_waimai_10k()
# exit(0)

clf = BertClassifier(bert_name='bert-base-chinese', label_number=2, max_sentence_len=100)
df = pd.read_csv('localdata/waimai5000.csv')
clf.fit(df.text, df.label, epochs=10, batch_size=32)

