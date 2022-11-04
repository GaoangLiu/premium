#!/usr/bin/env python
import tensorflow as tf
import os
import sys
import json
import re
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

# hub_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
# embed = hub.KerasLayer(hub_url)
# embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
from premium.crf import CRF
from premium.models.crf.preprocess import get_label_sbme, restore_from_label
import codefast as cf

xs = cf.io.read('/tmp/train')
xs = [cf.eval(i)[1] for i in xs]
xs = [get_label_sbme(i) for i in xs]
labels = [[_[1] for _ in x] for x in xs]

cli = CRF()
cli.fit(xs, labels)
cli.save_model('/tmp/price.model')


ts1 = cf.io.read('/tmp/test')
ts = [list(cf.eval(i)[1]) for i in ts1]
model = CRF.load_model('/tmp/price.model')
cf.info('model loaded')
tag_list = model.predict(ts)
for text, tags in zip(ts, tag_list):
    s = restore_from_label(text, tags)
    print(('customer_contact', s))
