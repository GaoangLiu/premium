#!/usr/bin/env python

import re
import argparse
from typing import List, Tuple, Dict, Set, Optional, Union, Callable
from crf import LinearChainCRF


class Config(object):
    MODEL_FILE = 'models/jieba-large.json'
    MODEL = LinearChainCRF(MODEL_FILE)

def _token(text: str) -> List[str]:
    tuples,res = zip(text, Config.MODEL.inference(text)),[]
    for a, b in tuples:
        if b in ('S', 'B'):
            res.append(a)
        else:
            if not res:res=['']
            res[-1] += a
    return res

def token(text: str) -> List[str]:
    """ Tokenize Chinese text into words, May need to seperate the text into a few
    sentences first.
    """
    tokens = []
    punctuations = set('，。！？')
    i, j = 0, 0
    while j <= len(text):
        if j == len(text) or text[j] in punctuations:
            tokens += _token(text[i:j+1])
            i = j+1
        j += 1
    return tokens        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", help="the model file name.")

    args = parser.parse_args()
    Config.MODEL = LinearChainCRF(args.modelfile)

    # crf.test(args.datafile)
    # exit(0)
    sent = '明天晚上吃什么，鱼和熊掌不可兼得'
    # sent = '喜欢秋天青岛的海，是蓝色的水银，地球的眼泪。像男子刚过中年，使不完的力气翻腾；更像待产的女人，已经有了母性的慈爱，忧郁而甘美。'
    # sent = '对是明白，那因为我们现在的话就是做那个灵活用功这一块的话，我们是免费开户的，就是六月份618的一个活动的话，我们是免费开户吗？想问一下，就是您也可以说就是先确定一个合作关系，然后我们就是先帮您去开户后七您有这个需求后再去走这个账，或者再去开这个发票，这样子都OK'
    # sent = '我这边只有三百万，不知道够不够'

    print(token(sent))
    import jieba
    print(jieba.lcut(sent))
