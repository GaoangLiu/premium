#!/usr/bin/env python
# Datasets manager.
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar

import codefast as cf
import numpy as np

from premium.utils import md5sum


class Urls(object):
    prefix = 'https://filedn.com/lCdtpv3siVybVynPcgXgnPm/corpus'


Path = TypeVar('Path', str, List[str])


def fetch_from_url(remote_url: str):
    target = f'/tmp/{cf.io.basename(remote_url)}'
    cf.info(f'downloading {remote_url}')
    cf.net.download(remote_url, target)
    # unzip data if necessary
    if target.endswith(('.zip', '.gz')):
        cf.info(f'Unzip file {target}')
        cf.shell(f'7z x {target} -o/tmp -y')


def fetch_data(fpath: str, sub_dir: str = None) -> None:
    cf.info(f'Downloading {fpath}')
    # support downloading model from third party
    if re.search('^http', fpath):
        online_url = fpath
    else:
        online_url = os.path.join(Urls.prefix, fpath)
        if sub_dir:
            online_url = cf.urljoin(Urls.prefix, sub_dir, fpath)
    fetch_from_url(online_url)


@dataclass
class DataFile(object):
    label: str
    name: str
    path: str
    md5sum: str = None
    description: str = None


class Math:

    @classmethod
    def get_coefs(cls, word, *arr):
        return word, np.asarray(arr, dtype='float32')


class WordToVector(object):

    def __init__(self) -> None:
        data_files = [
            DataFile(
                'glove-twitter-100-pickle',
                'glove.twitter.100d.pkl',
                'pretrained/glove.twitter.100d.pkl',
                md5sum='caa208506be76c61f1c5b36d2ac977d1',
                description='pickle file of glove-twitter-100 for faster loading'
            ),
            DataFile('glove-twitter-200-pickle',
                     'glove.twitter.200d.pkl',
                     'pretrained/glove.twitter.200d.pkl',
                     md5sum='3cc00e6be9a9895ff3d0e4b41898d509'),
            DataFile('tencent-embedding-100-small',
                     'tencent-embedding-100-small.pkl',
                     'pretrained/tencent-embedding-100-small.pkl'),
            DataFile('tencent-embedding-200-small',
                     'tencent-embedding-200-small.pkl',
                     'pretrained/tencent-embedding-200-small.pkl'),
            DataFile('glove-twitter-25', 'glove.twitter.27B.25d.txt',
                     'pretrained/glove.twitter.27B.25d.txt.gz'),
            DataFile('glove-twitter-50', 'glove.twitter.27B.50d.txt',
                     'pretrained/glove.twitter.27B.50d.txt.gz'),
            DataFile('glove-twitter-100',
                     'glove.twitter.27B.100d.txt',
                     'pretrained/glove.twitter.27B.100d.txt.gz',
                     md5sum='25f57e579da26f6b03b1fd4d8f04222f'),
            DataFile('glove-twitter-200',
                     'glove.twitter.27B.200d.txt',
                     'pretrained/glove.twitter.27B.200d.txt.zip',
                     md5sum='eb66f962466e34f0944060cdc435c559'),
            DataFile('glove-wiki-gigaword-50', 'glove.6B.50d.txt',
                     'pretrained/glove.6B.50d.txt'),
            DataFile('glove-wiki-gigaword-100', 'glove.6B.100d.txt',
                     'pretrained/glove.6B.100d.txt'),
            DataFile('glove-wiki-gigaword-200', 'glove.6B.200d.txt',
                     'pretrained/glove.6B.200d.txt'),
            DataFile('glove-wiki-gigaword-300', 'glove.6B.300d.txt',
                     'pretrained/glove.6B.300d.txt'),
            DataFile('google-news-negative-300',
                     'GoogleNews-vectors-negative300.bin',
                     'pretrained/GoogleNews-vectors-negative300.bin')
        ]
        self.vectors = dict((df.label, df) for df in data_files)
        self.data_files = data_files

    @property
    def list(self) -> list:
        return self.data_files

    def load_local(self, vector_file: str) -> dict:
        return dict(
            Math.get_coefs(*o.strip().split()) for o in cf.io.iter(vector_file))

    def load(self, vector: str = 'glove-twitter-25'):
        if vector not in self.vectors:
            cf.warning(f'{vector} was not here')
            return

        df = self.vectors[vector]
        file_name = f'/tmp/{df.name}'
        if cf.io.exists(file_name) and md5sum(file_name) == df.md5sum:
            cf.info('{} exists and is valid, skip downloading'.format(df.name))
        else:
            fetch_data(df.path)

        cf.info(f'loading {vector}')
        if file_name.endswith('.pkl'):
            return pickle.load(open(file_name, 'rb'))
        elif vector != 'google-news-negative-300':
            from gensim.models import KeyedVectors
            return KeyedVectors.load_word2vec_format(file_name)
        else:
            return KeyedVectors.load_word2vec_format(
                '/tmp/GoogleNews-vectors-negative300.bin', binary=True)


word2vec = WordToVector()


class Downloader(object):

    def twitter_disaster(self):
        """ Twitter real disaster or not """
        cf.info('Downloading twitter disaster data')
        fetch_data('twitter_disaster.csv', sub_dir='classification')

    def imdb(self):
        """ Englis imdb sentiment analysis 50000
        """
        cf.info('Downloading imdg sentiment dataset')
        fetch_data('imdb_sentiment.csv', sub_dir='classification')

    def douban_movie_review(self):
        '''Kaggle dataset https://www.kaggle.com/liujt14/dou-ban-movie-short-comments-10377movies
        size: 686 MB
        '''
        cf.info(
            'Downloading douban movie review data: https://www.kaggle.com/liujt14/dou-ban-movie-short-comments-10377movies',
        )
        fetch_data('douban_movie_review.zip')

    def douban_movie_review_2(self):
        fetch_data('douban_movie_review2.csv.zip')

    def chinese_mnist(self):
        '''https://www.kaggle.com/fedesoriano/chinese-mnist-digit-recognizer'''
        fetch_data('Chinese_MNIST.csv.zip')

    def toxic_comments(self):
        fetch_data('toxic_comments.csv')

    def icwb(self):
        '''Data source: http://sighan.cs.uchicago.edu/bakeoff2005/
        '''
        fetch_data('icwb2-data.zip')

    def news2016(self):
        ''' ???????????? 3.6 GB 2016????????? 
        '''
        fetch_data('news2016.zip')

    def msr_training(self):
        fetch_data('msr_training.utf8')

    def realty_roles(self):
        fetch_data('realty_roles.zip')
        cf.info('unzip files to /tmp/customer.txt and /tmp/salesman.txt')

    def realty(self):
        import getpass
        cf.info("Download real estate dataset realty.csv")
        zipped_data = os.path.join(Urls.prefix, 'realty.zip')
        cf.net.download(zipped_data, '/tmp/realty.zip')
        passphrase = getpass.getpass('Type in your password: ').rstrip()
        cf.utils.shell(f'unzip -o -P {passphrase} /tmp/realty.zip -d /tmp/')

    def spam_en(self):
        cf.info(f'Downloading English spam ham dataset to')
        fetch_data('spam-ham.txt')

    def spam_cn(self, path: str = '/tmp/'):
        cf.info(f'Downloading Chinese spam ham dataset to {path}')
        zipped_data = os.path.join(Urls.prefix, 'spam_cn.zip')
        label_file = os.path.join(Urls.prefix, 'spam_cn.json')
        cf.net.download(zipped_data, '/tmp/tmp_spam.zip')
        cf.utils.shell('unzip -o /tmp/tmp_spam.zip -d /tmp/')
        cf.net.download(label_file, '/tmp/spam_cn.json')

    def get_waimai_10k(self):
        # ????????????????????????????????????????????? 4000 ??????????????? 8000 ???
        fetch_data('waimai_10k.csv', 'classification')

    def get_online_shopping_10_cats(self):
        """10 ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? 6 ?????????????????????
        ???????????????????????? 3 ??????
        """
        fetch_data('online_shopping_10_cats.csv', 'classification')

    def chn_senti(self):
        """ ???????????????????????????. tag ???????????? train(9000), test(2000), dev(20000)
        tag,label,text
        train,1,????????????...
        test,1,....
        dev,1111
        """
        fetch_data('chn_senti_corp.csv', 'classification')

    def snli(self):
        fetch_from_url('https://host.ddot.cc/snli.zip')

    def mnli(self):
        """ Chinese-MNLI ???????????????????????????
        https://jishuin.proginn.com/p/763bfbd72666
        """
        fetch_from_url('https://host.ddot.cc/mnli.zip')

    def get(self, filename: str):
        url_map = {
            'tencent_cn_1M.txt':
            'https://filedn.com/lCdtpv3siVybVynPcgXgnPm/corpus/pretrained/tencent_cn/tencent_cn_1M.txt',
            'glove25d.gz':
            'https://filedn.com/lCdtpv3siVybVynPcgXgnPm/corpus/pretrained/glove.twitter.27B.25d.txt.gz'
        }
        if filename in url_map:
            cf.net.download(url_map[filename], '/tmp/' + filename)

downloader = Downloader()

