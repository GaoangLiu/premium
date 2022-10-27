#!/usr/bin/env python
import tensorflow as tf
import os, sys, json
import re
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

# hub_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
# embed = hub.KerasLayer(hub_url)
# embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
# print(embeddings.shape, embeddings.dtype)
import premium as pm

pm.datasets.mnli()

