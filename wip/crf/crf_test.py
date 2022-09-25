#!/usr/bin/env python

import argparse

from crf import LinearChainCRF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="data file for testing input")
    parser.add_argument("modelfile", help="the model file name.")

    args = parser.parse_args()

    crf = LinearChainCRF()
    crf.load(args.modelfile)
    # crf.test(args.datafile)
    # exit(0)
    sent = '明天晚上吃什么，鱼和熊掌不可兼得'
    sent = '喜欢秋天青岛的海，是蓝色的水银，地球的眼泪。像男子刚过中年，使不完的力气翻腾；更像待产的女人，已经有了母性的慈爱，忧郁而甘美。'
    # sent = '今天天气不错，我准备一会儿打辆车回去了，但是可能会堵车'
    sent = '对是明白，那因为我们现在的话就是做那个灵活用功这一块的话，我们是免费开户的，就是六月份618的一个活动的话，我们是免费开户吗？想问一下，就是您也可以说就是先确定一个合作关系，然后我们就是先帮您去开户后七您有这个需求后再去走这个账，或者再去开这个发票，这样子都OK'
    for _ in range(1):
        resp = crf.inference(sent)
    # print(resp)
    # exit(0)
    tuples = zip(sent, resp)
    res = []
    for a, b in tuples:
        if b == 'S':
            res.append(a)
        elif b == 'B':
            res.append(a)
        else:
            res[-1] += a
    print(res)
    import jieba
    print(jieba.lcut(sent))
    
    