import math
import pickle
from collections import defaultdict
from typing import List, Tuple

import codefast as cf


class HMM:
    def __init__(self):
        cur_dir = io.dirname()
        self.TRAIN_CORPUS = f'{cur_dir}/data/msr_training.utf8'
        self.emit_pickle = f'{cur_dir}/data/emit_p.pickle'
        self.trans_pickle = f'{cur_dir}/data/trans.pickle'
        self.punctuations = set('，‘’“”。！？：（）、')
        self.MIN_FLOAT = -1e10
        self.emission = None
        self.transition = None

    def train(self) -> Tuple[dict, list]:
        ''' train HMM segment model from corpus '''
        if self.emission and self.transition:     # to avoid I/O
            return self.emission, self.transition

        if io.exists(self.emit_pickle) and io.exists(self.trans_pickle):
            emit_p = pickle.load(open(self.emit_pickle, 'rb'))
            trans = pickle.load(open(self.trans_pickle, 'rb'))
            self.emission, self.transition = emit_p, trans
            return emit_p, trans

        emit_p = defaultdict(int)
        ci = defaultdict(int)
        # b0 m1 e2 s3
        trans = [[0] * 4 for _ in range(4)]

        def update_trans(i, j):
            if i < 0:
                return j
            trans[i][j] += 1
            return j

        with open(self.TRAIN_CORPUS, 'r') as f:
            for ln in f.readlines():
                pre_symbol = -1
                _words = (e for e in ln.split(' ') if e)
                for cur in _words:
                    if cur in self.punctuations:
                        pre_symbol = -1
                    else:
                        if len(cur) == 1:
                            emit_p[(cur, 'S')] += 1
                            ci['S'] += 1
                            pre_symbol = update_trans(pre_symbol, 3)
                        else:
                            for i, c in enumerate(cur):
                                if i == 0:
                                    emit_p[(c, 'B')] += 1
                                    ci['B'] += 1
                                    pre_symbol = update_trans(pre_symbol, 0)
                                elif i == len(cur) - 1:
                                    emit_p[(c, 'E')] += 1
                                    ci['E'] += 1
                                    pre_symbol = update_trans(pre_symbol, 2)
                                else:
                                    ci['M'] += 1
                                    emit_p[(c, 'M')] += 1
                                    pre_symbol = update_trans(pre_symbol, 1)
        cf.info('count pairs complete.')

        for i, t in enumerate(trans):     # normalization
            trans[i] = [
                math.log(e / sum(t)) if e > 0 else self.MIN_FLOAT for e in t
            ]

        for key, v in emit_p.items():
            emit_p[key] = math.log(v / ci[key[1]])

        with open(self.emit_pickle, 'wb') as f:
            pickle.dump(emit_p, f)

        with open(self.trans_pickle, 'wb') as f:
            pickle.dump(trans, f)

        return emit_p, trans

    def calculate_trans(self, emit_p: dict, ci: dict, obs: str,
                        state: str) -> float:
        return (1 + emit_p.get((obs, state), 0)) / ci.get(state, 1)

    def viterbi(self, text: str, emit_p: dict, trans_p: dict) -> List[str]:
        ''' a DP method to locate the word segment scheme with maximum probability 
        for a given Chinese sentence.
        :param text: str, observed sequence, e.g., '人性的枷锁'
        :param emit_p: dict, emission probability matrix, e.g., emit_p[('一', 'B')] = 0.0233
        :param trans_p: dict, transition probability matrix, e.g., trans_p['B']['M'] = 0.123
        :return: list[str], word segments, e.g., ['人性', '的' ,'枷锁']
        '''
        if not text:
            return []
        state_index = dict(zip('BMES', range(4)))
        cache = {}

        for i, c in enumerate(text):
            if i == 0:
                for s in 'BS':
                    cache[s] = (-0.5, s)     # this initial prob is customizable
                cache['E'] = (self.MIN_FLOAT, 'E')
                cache['M'] = (self.MIN_FLOAT, 'M')
            else:
                cccopy = cache.copy()
                for s in 'BMES':
                    max_prob, prev_seq = float('-inf'), ''
                    for prev_state, v in cccopy.items():
                        prev_index, cur_index = state_index[
                            prev_state], state_index[s]

                        # not '*' but '+'
                        new_prob = v[0] + trans_p[prev_index][
                            cur_index] + emit_p.get((c, s), self.MIN_FLOAT)
                        if new_prob > max_prob:
                            max_prob = new_prob
                            prev_seq = v[1]
                    cache[s] = (max_prob, prev_seq + '->' + s)

        # assume a sentence ends with either 'E' or 'S'
        # print(cache)
        seq = cache['E'][1] if cache['E'][0] > cache['S'][0] else cache['S'][1]
        return self._cut(text, seq)

    def _cut(self, text: str, seq: str) -> List[str]:
        ''' seperate a sequence by word lexeme.
        e.g., input ('人性的枷锁', 'B->E->S->B->E'), output ['人性','的','枷锁']
        '''
        res = []
        # print(seq, text)
        # print(len(text))
        for a, b in zip(text, seq.split('->')):
            if (b == 'B' or b == 'S'):
                res.append('')
            res[-1] += a
            # print(res)

        return res

    def segment(self, text: str) -> List[str]:
        emit_p, trans = self.train()
        res, prev_text = [], ''
        for i, c in enumerate(text):
            if c in self.punctuations:
                res += self.viterbi(prev_text, emit_p, trans)
                res.append(c)
                prev_text = ''
            else:
                prev_text += c

        if prev_text:
            res += self.viterbi(prev_text, emit_p, trans)
        return res


if __name__ == '__main__':
    cur_dir = io.dirname()
    io.rm(f'{cur_dir}/data/emit_p.pickle')
    texts = [
        '看了你的信，我被你信中流露出的凄苦、迷惘以及热切求助的情绪触动了。', '这是一种基于统计的分词方案', '这位先生您手机欠费了',
        '还有没有更快的方法', '买水果然后来世博园最后去世博会', '欢迎新老师生前来就餐', '北京大学生前来应聘', '今天天气不错哦',
        '就问你服不服', '我们不只在和你们一家公司对接', '结婚的和尚未结婚的都沿海边去了', '这也许就是一代人的命运吧',
        '改判被告人死刑立即执行', '检察院鲍绍坤检察长', '腾讯和阿里都在新零售大举布局', '人性的枷锁'
    ]

    texts += [
        '好的，现在我们尝试一下带标点符号的分词效果。', '中华人民共和国不可分割，坚决捍卫我国领土。',
        '英国白金汉宫发表声明，菲利浦亲王去世，享年九十九岁。', '扬帆远东做与中国合作的先行', '不是说伊尔新一轮裁员了。',
        '滴滴遭调查后，投资人认为中国科技业将强化数据安全合规。','小明硕士毕业于中国科学院计算所，后在日本京都大学深造'
    ]
    for text in texts:
        ret = ' '.join(HMM().segment(text))
        print(ret)
