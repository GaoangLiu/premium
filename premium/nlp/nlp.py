#!/usr/bin/env python
import re
from itertools import product

import codefast as cf


class ner:
    def __init__(self):
        ...

    def extract_company(self, text: str) -> list:
        ...


'''TODO 
什么时间回家。
'''


def regex_inquiry(text: str) -> bool:
    pats = []
    pats.append(re.compile(r'^什么(时间|时候|形式).{0,3}$'))
    pats.append(re.compile(r'不是说.*(了|吗)'))
    excludes = []
    return any(re.search(p, str(text)) for p in pats)


def is_question(text: str) -> bool:
    # cf.info(f'input sentence: {text}')
    strong_modal = ('吗', '吧', '嘛')
    strong_punctuations = ('?', '？')
    _weak_modal = ('呢', '没有', '不')
    _weak_punctuations = ('。', '.')

    if any(e in text for e in (strong_modal + strong_punctuations)):
        return True

    return False


if __name__ == '__main__':
    text = [
        '行，那行，已经到家了是嘛。', '你大概准备什么时间去美国打工', '我不清楚你说的是什么时间的事也不想知道。', '什么时间回家',
        '不是说伊尔新一轮裁员了。'
    ]

    for t in text:
        print(t, is_question(t))
        print(t, regex_inquiry(t))
