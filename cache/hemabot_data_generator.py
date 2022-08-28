#!/usr/bin/env python
"""
A classifier for hemabot messages. To classify messages, 
we use a machine learning model to predict the message type.
"""
import datetime
import random
from typing import List

import codefast as cf
import pandas as pd
from faker import Faker

faker = Faker()

KEYWORDS_MAP = [
    ('oncemessage', ['oncemessage', 'onecemesage', 'oncemsg', 'oncemassage']),
    ('weather', [
        'weather',
        'how is the weather',
        'how is weather',
        "how's weather",
        "what is the weather",
        "what's weather",
        'tainqi',
        'tianqi',
        'tianqiyubao',
    ]),
    ('avatar', [
        'avadar', 'avatar', 'touxiang', 'headiamge', 'headimage', 'head iamge',
        'avatarimg', 'avatarimgs'
    ]), ('deepl', ['deepl', 'deelp', 'translate', 'deeptranslate']),
    ('pcloud', ['pcloud', 'pclouduplaod', 'pcloudupload']),
    ('twitter', ['twitter.com'])
]


def split_url(url: str) -> str:
    for punc in ':?=&.(),':
        url = url.replace(punc, '/')
    return ' '.join([e for e in url.split('/') if e])


def generate_twitter_url(user_name: str) -> str:
    # Generate twitter status url
    now = datetime.datetime.now()
    days_shift = random.randint(0, 1200)
    old_day = now - datetime.timedelta(days=days_shift)
    timestamp = str(old_day.timestamp()).replace('.', '') + str(
        random.randint(100, 999))
    url = 'https://twitter.com/{}/status/{}'.format(user_name, timestamp)
    return split_url(url)


def generate_pcloud_url() -> str:
    hostname = faker.hostname()
    if random.randint(0, 10) > 3:  # add noise
        user = faker.user_name()
        hostname = generate_twitter_url(user).replace('https://', '')
    http = 'http ' if random.randint(0, 1) == 0 else 'https '
    file_path = faker.file_path(category='video')
    x_len = random.randint(10, 30)
    extra_args = "{}".format(cf.random_string(
        length=x_len)) if random.randint(0, 1) == 1 else ""
    url = cf.urljoin(http, hostname, file_path, extra_args)

    return split_url(url)


def generate_deepl(count: int = 10) -> List[str]:
    res = []
    for i in range(count):
        nb_sentences = random.randint(1, 10)
        content = faker.paragraph(nb_sentences=nb_sentences,
                                  variable_nb_sentences=True)
        res.append('{}'.format(content))
    return res


xs = generate_deepl(1000)
for x in xs:
    print('deepl,{}'.format(x))


def generate_oncemessage(count: int = 100) -> str:
    # Generate oncemessage
    class TimeFormat(object):
        def __init__(self):
            self.seprators = ['-', ':', ' ', '/']

        def seconds(self):
            return str(random.randint(0, 1 << 15))

        def hour_minute(self):
            return '{}{}{}'.format(random.randint(0, 23),
                                   random.choice(self.seprators),
                                   random.randint(0, 59))

        def date_time(self):
            sep = random.choice(self.seprators)
            return '{}{}{} {}{}{}{}{}'.format(random.randint(1, 12), sep,
                                              random.randint(1, 31),
                                              random.randint(0, 23), sep,
                                              random.randint(0, 59), sep,
                                              random.randint(0, 59))

    old_timer = TimeFormat()
    time_stratergies = [
        old_timer.seconds, old_timer.hour_minute, old_timer.date_time
    ]
    res = []
    hints = ['oncemessage', 'onecemesage', 'oncemsg', 'oncemassage']
    for _ in range(count):
        hint = random.choice(hints)
        time_stratergy = random.choice(time_stratergies)
        content = faker.paragraph(nb_sentences=1, variable_nb_sentences=True)
        res.append('{} {} {}'.format(hint, time_stratergy(), content))
    return res


def add_sample(row: pd.Series) -> str:
    # create hemabot classifier data based on twitter sentiment
    # dataset
    text, key = row['tweet'], row['target']
    pair = next((p for p in KEYWORDS_MAP if key == p[0]), None)
    assert pair is not None, "keyword not found"

    if key == 'deepl':
        return generate_deepl(text)

    elif key == 'weather' or key == 'avatar':
        return random.choice(pair[1])

    elif key == 'twitter':
        return generate_twitter_url(row['user'])

    elif key == 'pcloud':
        return generate_pcloud_url()

    elif key == 'oncemessage':
        words = text.split(' ')
        msg_kw = random.sample(pair[1], 1)[0]
        words.insert(0, msg_kw)
        return ' '.join(words)

    else:
        raise Exception('unknown key:', key)


def ftext():
    df = pd.read_csv('/tmp/10k.csv', encoding="ISO-8859-1")
    keys = [k for k, _ in KEYWORDS_MAP]
    labels = [random.choice(keys) for _ in range(len(df))]
    df['target'] = labels
    df['text'] = df.apply(add_sample, axis=1)

    df = df[['target', 'text']]
    df.to_csv('localdata/tmp.csv', index=False)
    print(df)


def try_bert():
    df = pd.read_csv('/tmp/10k.csv', encoding="ISO-8859-1")
    keys = [k for k, _ in KEYWORDS_MAP]
    labels = [random.choice(keys) for _ in range(len(df))]
    df['target'] = labels
    df['text'] = df.apply(add_sample, axis=1)

    df = df[['target', 'text']]
    df.to_csv('localdata/tmp.csv', index=False)
    # from premium.models.bert import bert_benchmark
    # bc = bert_benchmark(df, epochs=2)

    test_texts = [
        'this is a quite short sentence.', 'oncemssage 12:03 somerurn',
        'avatar', 'www baidu com video mp4'
    ]
    test_texts += cf.io.read('localdata/test_hema.txt')
    bc = BertClassifier(bert_name='distilbert-base-uncased',
                        num_labels=6,
                        weights_path='/tmp/logs/keras.h5')

    xs = bc.predict(test_texts)
    label_map = cf.js('/tmp/label_map.json')
    label_map = dict((v, k) for k, v in label_map.items())
    for i, text in enumerate(test_texts):
        print(label_map[xs[i]], text)
