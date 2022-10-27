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
from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('bert-base-chinese')

# Two lists of sentences
sentences1 = ['技术现实主义将被历史视为一场悲剧性的运动。',
              '我脑子里一直在想一个主意，我觉得现在该提出来了。',
              '是啊，这几天听起来像约翰尼。']

sentences2 = ['他们被这场运动弄得精神上伤痕累累。', '我对告诉他们犹豫不决。', '这几天听起来像约翰尼']

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print("{:<30} {:<30} Score: {:.4f}".format(
        sentences1[i], sentences2[i], cosine_scores[i][i]))
