#!/usr/bin/env python

import argparse
from typing import List, Tuple, Dict, Set, Optional, Union, Callable
from crf import LinearChainCRF
import codefast as cf


class Config(object):
    MODEL_FILE = 'models/money.json'
    MODEL = LinearChainCRF(MODEL_FILE)


def ner(text: str) -> List[str]:
    resp = Config.MODEL.inference(text)
    tuples = list(zip(text, resp))
    res = []
    for a, b in tuples:
        if b == 'O':
            res.append('')
        else:
            if not res:
                res = ['']
            res[-1] += a
    return tuples, [e for e in res if e]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", help="the model file name.")

    args = parser.parse_args()
    Config.MODEL = LinearChainCRF(args.modelfile)

    text = '我这边只有三百万，加上你的有1200块，加上昨天的七千六百四十三元，不知道够不够'
    # text = '2000多，然后2000加上一个地址，反正也就6000多块钱吧，好像。'
    text='对呀，那其实我们现在我们现在活动价格零申报也挺便宜的，就比你少个200块钱咯'
    for _ in range(1):
        ner(text)
    print(text, ner(text))
    import pickle 
    js = cf.js(args.modelfile)
    pickle.dump(js, open('/tmp/x.pkl', 'wb'))

