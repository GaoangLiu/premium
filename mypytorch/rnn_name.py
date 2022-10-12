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
from rich import print
# from premium.data.datasets import dataloader


def load_data() -> Dict:
    url = 'https://f004.backblazeb2.com/file/b2c3d6/tmp/rnnnames.json'
    # return cf.l(cf.io.walk('/tmp/names')).map(
    #     lambda f: (cf.io.stem(f), cf.io.read(f).tolist())).to_dict()
    cf.net.download(url, '/tmp/rnnnames.json')
    return cf.js('/tmp/rnnnames.json')


dt = load_data()
print(dt['Italian'][:5])
