#!/usr/bin/env python
from collections import defaultdict
from functools import reduce
from typing import List, Union

import codefast as cf
import json
import requests


class const(object):
    # Tencent cloud function endpoint
    tcf_endpoint = 'https://service-argadd49-1303988041.bj.apigw.tencentcs.com/release/'
    nlp_endpoint = 'https://service-argadd49-1303988041.bj.apigw.tencentcs.com/release/nlp'


def tok_jieba(inputs: Union[str, List[str]]) -> List[str]:
    _json = {
        'task_type': 'token',
        'sentence': inputs
    } if isinstance(inputs, str) else {
        'task_type': 'batch_token',
        'sentence_list': inputs
    }
    return requests.post(const.nlp_endpoint, json=_json).json()


def word_embedding(inputs: Union[str, List[str]]) -> List[float]:
    _json = {
        'task_type': 'word_embedding',
        'word': inputs
    } if isinstance(inputs, str) else {
        'task_type': 'batch_word_embedding',
        'word_list': inputs
    }
    text = requests.post(const.nlp_endpoint, json=_json,timeout=30).text
    try:
        return json.loads(text)
    except:
        cf.warning('word_embedding failed: {}'.format(text))
        return {"result": text}
        
