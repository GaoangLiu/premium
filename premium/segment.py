import os
from typing import List

import codefast as cf


class TestData:
    sentences = [
        '看了你的信，我被你信中流露出的凄苦、迷惘以及热切求助的情绪触动了。', '这是一种基于统计的分词方案', '这位先生您手机欠费了',
        '还有没有更快的方法', '买水果然后来世博园最后去世博会', '欢迎新老师生前来就餐', '北京大学生前来应聘', '今天天气不错哦',
        '就问你服不服', '我们不只在和你们一家公司对接', '结婚的和尚未结婚的都沿海边去了', '这也许就是一代人的命运吧',
        '改判被告人死刑立即执行', '检察院鲍绍坤检察长', '腾讯和阿里都在新零售大举布局', '人性的枷锁',
        '好的，现在我们尝试一下带标点符号的分词效果。', '中华人民共和国不可分割，坚决捍卫我国领土。',
        '英国白金汉宫发表声明，菲利浦亲王去世，享年九十九岁。', '扬帆远东做与中国合作的先行', '不是说伊尔新一轮裁员了。',
        '滴滴遭调查后，投资人认为中国科技业将强化数据安全合规。', '小明硕士毕业于中国科学院计算所，后在日本京都大学深造。',
        '他说明天王小丫上班肯定迟到。',
        '根据签订的金融合作协议，国家开发银行将在“十五”期间为北京市城市轻轨、高速公路、公路联络线、信息网络、天然气、电力、大气治理和水资源环保等提供人民币500亿元左右的贷款支持；同意三年内为中关村科技园区建设提供人民币总量80亿至100亿元的贷款支持；对推荐的项目优先安排评审，实行评审“快通道”，对信用等级较高的企业将给予一定的授信；通过参与企业资本运作，防范信贷风险；受北京市政府委托为首都制定经济结构调整规划、整合创新资源、培育风险投资体系、产业结构调整和升级、重大项目决策和可行性研究、项目融资、银团贷款和资产重组等提供金融顾问服务等。北京市政府将积极帮助协调开发银行贷款项目的评审和管理、本息回收、资产保全等，促进开发银行信贷风险的防范和化解。',
        '顶住压力。',
        '《代表法》等法律法规规定，在人民代表大会闭会期间，各级人大代表要与原选举单位保持密切的联系，听取和反映人民的意见；要开展视察，进行调查研究，反映群众的呼声和要求。在组织形式上，一是按要求参加人大常委会或乡（镇）人大主席团组织的活动，如视察、调查、检查、评议、列席有关会议等；二是积极参加人大常委会或乡（镇）人大主席团协助组织的代表小组的活动；三是可以由几名代表自行联名或代表个人持代表证就地进行视察。'
    ]


class MMM():
    """Maximum Matching Model，最大匹配法 """
    def __init__(self):
        self._words = set()
        self.load_dictionary()

    def load_dictionary(self):
        file_path = os.path.join(cf.io.dirname(), 'localdata')

        with open(f'{file_path}/dict.txt') as fd, open(
                f'{file_path}/oov.txt') as fo:
            for line in fd.readlines() + fo.readlines():
                line = line.replace('\t', ' ')
                self._words.add(line.split(' ')[0].rstrip())

    def forward_segment(self, text: str) -> List:
        word_list = []
        i = 0
        while i < len(text):
            longest_word = text[i]
            for j in range(i + 1, len(text)):
                word = text[i:j]
                if word in self._words:
                    if len(word) > len(longest_word):
                        longest_word = word
            word_list.append(longest_word)
            i += len(longest_word)
        return word_list

    def backward_segment(self, text: str) -> List:
        word_list = []
        i = len(text) - 1
        while i >= 0:
            longest_word = text[i]
            for j in range(i):
                word = text[j:i + 1]
                if word in self._words:
                    if len(word) > len(longest_word):
                        longest_word = word
                        break
            word_list.insert(0, longest_word)
            i -= len(longest_word)
        return word_list

    def count_single_char(self, word_list: List) -> int:  # 统计单字成词的个数
        return sum(1 for word in word_list if len(word) == 1)

    def bidirectional_segment(self, text: str) -> List[str]:
        """双向最大匹配"""
        f = self.forward_segment(text)
        b = self.backward_segment(text)
        if len(f) < len(b):  # 词数更少优先级更高
            return f
        elif len(f) > len(b):
            return b
        else:
            if self.count_single_char(f) < self.count_single_char(
                    b):  # 单字更少优先级更高
                return f
            else:
                return b


if __name__ == '__main__':
    model = MMM()
    text = '好的。凭您的手机号是微信吗？我稍稍后加一下你吧，把相关的资料发给您。'
    for s in TestData.sentences:
        # print(model.forward_segment(text))
        print(model.bidirectional_segment(s))
