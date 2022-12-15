#!/usr/bin/env python3
import codefast as cf

import numpy as np
import pandas as pd

from typing import List, Union, Callable, Set, Dict, Tuple, Optional
import torch


def get_device()->torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

