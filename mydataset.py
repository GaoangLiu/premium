#!/usr/bin/env python
import random
import re, os, sys, joblib
from collections import defaultdict
from functools import reduce
import codefast as cf

import numpy as np
import torch as T

class IMDB_Dataset(T.utils.data.Dataset):
  # each line: 20 token IDs, 0 or 1 label. space delimited
  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(0,21),
      delimiter=" ", comments="#", dtype=np.int64)
    tmp_x = all_xy[:,0:20]   # cols [0,20) = [0,19]
    tmp_y = all_xy[:,20]     # all rows, just col 20
    self.x_data = T.tensor(tmp_x, dtype=T.int64) 
    self.y_data = T.tensor(tmp_y, dtype=T.int64)  # CE loss

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    token_ids = self.x_data[idx]
    trgts = self.y_data[idx] 
    return (token_ids, trgts)

    