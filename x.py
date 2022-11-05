#!/usr/bin/env python
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
from rich import print

from bner import BertModel, Cfg, NerModel, Per
from premium.data.datasets import downloader
from premium.models.bert import BertClassifier

model = BertModel()
model.load_state_dict(torch.load(Cfg.model_path))
cf.info('model loaded')
NerModel.evaluate(model, Per.dataset.test)
