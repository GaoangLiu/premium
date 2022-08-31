#!/usr/bin/env python
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
from rich import print

from premium.experimental.cbow import SimpleCBOW

words_list = [['are', 'you', 'okay'], ['im', 'okay', 'you', 'seem', 'okay']]

cbow = SimpleCBOW(words_list)
cbow.tokenize()
# for x, y in cbow.generate_context_word_pairs(cbow.word_ids):
#     print(x, y)
cbow.train()
