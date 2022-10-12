#!/usr/bin/env python
import random
import re
import os
import sys
import joblib
from collections import defaultdict
from functools import reduce
import codefast as cf

from premium.data.datasets import downloader

from premium.experimental.utils.noise import add_gaussian_noise, gaussian

# add_gaussian_noise('/tmp/pepper.png', '/tmp/x.png')
gaussian('/tmp/pepper.png', '/tmp/x.png', .1)
