#!/usr/bin/env python
import random
import re
import os
import sys
import joblib
from collections import defaultdict
from functools import reduce
import codefast as cf
from premium.metrics.distance import jaccard, wmd

a = set([1, 2, 3])
b = set([2, 3, 4])
print(jaccard(a, b))

from gensim.models import KeyedVectors

model_path = '/data/pretraining/tencent100.txt'
model = KeyedVectors.load_word2vec_format(model_path, binary=False)
cf.info('model loaded')
s1 = ['我', '喜欢', '你']
s2 = ['我', '不', '喜欢', '你']
s3 = ['我', '爱', '你']
for _ in range(1000):
    dis = model.wmdistance(s1, s2)
    dis2 = model.wmdistance(s1, s3)
print(dis, dis2)
cf.info('done')
exit(0)

X, y = [[]], [[]]
for l in cf.io.read('/tmp/token_train'):
    if len(l.strip()) == 0:
        X.append([])
        y.append([])
    else:
        try:
            a, b = l.strip().split()
            X[-1].append(a)
            y[-1].append(b)
        except:
            pass
X = X[:100000]
y = y[:100000]
# data = cf.io.read('/tmp/tmp.txt').map(get_label).data
# X = [x for x, _ in data]
# y = [y for _, y in data]
print(random.sample(y, 10))
m = CRF(X, y)
m.fit()

test = [
    list('你们用什么语言？Java还是Python ++，前端开发就是做账吗？'),
    list('底盘的话一般用这个业务，一般用java和python，然后目前的话java去做。')
]
y = m.predict(test)
print(y)
