#!/usr/bin/env python
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from functools import reduce
from timeit import timeit
from typing import Dict, List, Optional, Set, Tuple

import codefast as cf
import joblib
import pandas as pd
import tensorflow as tf
from codefast.decorators.log import time_it
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers

from premium.preprocessing.text import TextVectorizer

ds = load_dataset('conll2003')
print(ds['train'][0])
