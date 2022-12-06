#!/usr/bin/env python3
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
from premium.utils import cf_unless_tmp

cf_unless_tmp('vehicle.csv')
d = pd.read_csv('/tmp/vehicle.csv')
d.drop(columns=[
    '行标签', '用户好评率（9.4前两周平均）', '9月4日健康车周转', '累积订单比(可以不看）', '9月4日健康车数量',
    '9月4日电单车有效骑行订单数'
],
       inplace=True)
print(d.columns.tolist())
# d = d[['供需', '竞争', '用户好评率（9.4前两周平均）']]
features = d.columns.tolist()
dtypes = dict(d.dtypes)
print(dtypes)
for f in features:
    if len(d[f].value_counts()) == 1:
        d.drop(columns=[f], inplace=True)
        continue

    if dtypes[f] == 'object':
        d[f] = d[f].fillna('OO')
        dummy = pd.get_dummies(d[f], prefix=f)
        d.drop(columns=[f], inplace=True)
        d = pd.concat([d, dummy], axis=1)
    elif dtypes[f] == 'int64':
        d[f] = d[f].fillna(0)
    elif dtypes[f] == 'float64':
        avg = d[f].mean()
        d[f] = d[f].fillna(avg)
    else:
        raise Exception(f'Unknown type: {dtypes[f]}')

print(d)

from sklearn.preprocessing import StandardScaler

print(d.columns.tolist())
d = StandardScaler().fit_transform(d)
print(d)

X = d
import matplotlib.pyplot as plt
import numpy as np
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
from sklearn.cluster import MiniBatchKMeans

model = MiniBatchKMeans(n_clusters=2)
# 模型拟合
model.fit(X)
# 为每个示例分配一个集群
yhat = model.predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    # 创建这些样本的散布
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
pyplot.show()
