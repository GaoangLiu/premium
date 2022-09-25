#!/usr/bin/env python

import argparse
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf

from crf import LinearChainCRF


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

    texts = [
        '没听明白，是这样子的，就比如说我们的工资，一个月是2300块钱吗。', '平均下来一个月的话就是400多块钱一个月吗。',
        '我这边只有三百块，加上你的有1200块，加上昨天的七千六百四十三元，不知道够不够',
        '2000多，然后2000加上一个地址，反正也就6000多块钱吧，好像。',
        '然后现在两年的话，它是有活动价9988，平均下来一个月的话就是400多块钱一个月吗。',
        '如果说是免这免免那个个人所得税额度是在6万，现在是每个人对每个人的那个年度所得，就是每个月5000嘛，对不对，一年就是6万块吗？超了6万块之后，超出额度是按七级去缴纳它的一个个人所得税的。',
        '没有的话，现在618出了一个特定的套餐，比之前给您报的那个价格便宜。', '我们一年是2000吗', '你两年就是三千九百九一年',
        '价格是三千九百九一年'
    ]
    
    for text in texts:
        print(f">> Input: {text}", f"\nResults: {ner(text)[1]}\n")
    # print(text, ner(text))
