#!/usr/bin/env python
"""
A classifier for hemabot messages. To classify messages, 
we use a machine learning model to predict the message type.
"""
import datetime
import random

import codefast as cf
import fasttext
import pandas as pd
from faker import Faker

from premium.experimental.myfasttext import benchmark as ftb
from premium.models.bert import BertClassifier

faker = Faker()

KEYWORDS_MAP = [('oncemessage',
                 ['oncemessage', 'onecemesage', 'oncemsg', 'oncemassage']),
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
                    'avadar', 'avatar', 'touxiang', 'headiamge', 'headimage',
                    'head iamge', 'avatarimg', 'avatarimgs'
                ]), ('deepl', ['deepl', 'deelp', 'translate', 'deeptranslate']),
                ('pcloud', ['pcloud', 'pclouduplaod', 'pcloudupload']),
                ('twitter', ['twitter.com'])]


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
    if random.randint(0, 10) > 3:     # add noise
        user = faker.user_name()
        hostname = generate_twitter_url(user).replace('https://', '')
    http = 'http ' if random.randint(0, 1) == 0 else 'https '
    file_path = faker.file_path(category='video')
    x_len = random.randint(10, 30)
    extra_args = "{}".format(cf.random_string(
        length=x_len)) if random.randint(0, 1) == 1 else ""
    url = cf.urljoin(http, hostname, file_path, extra_args)

    return split_url(url)


def generate_deepl(text: str) -> str:
    if random.randint(0, 1) == 0:
        return text
    else:
        words = text.split(' ')
        rand_index = random.randint(1,
                                    len(words) -
                                    1)     # insert after first word
        # Reduce confusion with weather labels and onemessage labels
        if random.randint(0, 1) == 0:
            weather_list = KEYWORDS_MAP[1][1]
            noise = random.choice(weather_list)
        else:
            message_list = KEYWORDS_MAP[0][1]
            noise = random.choice(message_list)
        words.insert(rand_index, noise)
        if random.randint(0, 10) == 1:
            return random.choice(['how are you', 'are you okay'])
        return ' '.join(words)


def add_sample(row: pd.Series) -> str:
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


try_bert()
exit(0)

model_path = 'localdata/hema_model.bin'
model = fasttext.load_model(model_path)
for text in cf.io.read('localdata/test_hema.txt'):
    text = split_url(text)
    msg = {'text': text, 'label': model.predict(text)}
    print(msg)
