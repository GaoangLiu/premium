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

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(version_base=None, config_path='data/config/', config_name='demo.yaml')
def test(cfg):
    print(cfg)
    # print(cfg.db.redis)

if __name__ == "__main__":
    test()