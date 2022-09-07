#!/usr/bin/env python
import os
import random
import re
import sys
import time
from collections import defaultdict
from functools import reduce
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import joblib
import numpy as np
import pandas as pd
from authc import gunload
from rich import print


class DeeplAPI(object):
    '''Deepl tranlation API'''

    def __init__(self) -> None:
        self._url = 'https://api-free.deepl.com/v2'
        self._headers = '''Host: api-free.deepl.com
            User-Agent: YourApp
            Accept: */*
            Content-Length: [length]
            Content-Type: application/x-www-form-urlencoded'''
        self._token = gunload('deepl_token')
        self._params = {'auth_key': self._token}

    def do_request(self, api_path: str) -> dict:
        resp = cf.net.post(self._url + api_path,
                           headers=cf.net.parse_headers(self._headers),
                           data=self._params)
        if resp.status_code != 200:
            raise Exception(resp)
        return resp.json()

    @property
    def stats(self):
        return self.do_request('/usage')

    def translate(self, text: str) -> str:
        target_lang = 'EN' if cf.nstr(text).is_cn() else 'ZH'
        self._params['text'] = text
        self._params['target_lang'] = target_lang
        return self.do_request('/translate')


def back_translation(text: str) -> str:
    api = DeeplAPI()
    try:
        _text = api.translate(text)['translations'][0]['text']
        text = api.translate(_text)['translations'][0]['text']
        return text
    except Exception as e:
        return None


def back_translation_file(input_file_path: str, merged_file_path: str):
    """ Input a csv file, augument data by back translation and 
    export merged result to another csv file
    """
    df_input = pd.read_csv(input_file_path)
    x = df_input.copy()
    x['text'] = x.text.apply(back_translation)
    df_merged = pd.concat([df_input, x])
    df_merged.to_csv(merged_file_path, index=False)


if __name__ == '__main__':
    text_list = [
        '你如果是这样子的话，你七月份已经报了考试，你五月份六月份连着数学，然后看看七月份我们的考试。', '这么尴尬，四月27是吗。',
        '然后，你打算暑假考，那就已经七八月份了，然后九月份你肯定没办法考试的，一方面是绝对是雅思口语的换题季'
    ]

    for text in text_list:
        _text = back_translation(text)
        print(text, _text)
