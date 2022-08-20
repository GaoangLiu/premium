import os
import re
from abc import abstractmethod
from argparse import Namespace
from typing import Any, Dict, List, NamedTuple, Tuple

import codefast as cf
import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codefast.cn import strip_punc
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from premium.data._dict import contraction_dict

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()


class AbstractDataLoader(object):
    def __init__(self,
                 local_dir: str,
                 remote_dir: str,
                 files: List[str] = None):
        from dofast.file.syncfile import SyncFile
        demo = SyncFile('demo', local_dir=local_dir, remote_dir=remote_dir)
        self.syncfiles = list(map(demo.clone, files))

    @abstractmethod
    def __call__(self, frac=1.0):
        pass


class Corpus(NamedTuple):
    # x: train data, y: train label, test: test data, sub: submission data
    x: pd.DataFrame
    y: pd.Series = None
    t: pd.DataFrame = None
    s: pd.DataFrame = None

    def pie(self, df: pd.DataFrame, col: str) -> 'Corpus':
        # Draw pie chart
        pie, ax = plt.subplots(figsize=[18, 8])
        df.groupby(col).size().plot(kind='pie',
                                    autopct='%.2f',
                                    ax=ax,
                                    title='{} distibution'.format(col),
                                    cmap="Pastel1")
        plt.show()
        return self


class EDA(object):
    @classmethod
    def basic(cls, df):
        '''size, missing value count'''
        _info = {'Length': len(df), 'Missing count': df.isnull().sum()}
        for key, value in _info.items():
            print(key, value)


class TextChinese(object):
    def __init__(self, text: str):
        self.content = text

    @property
    def len(self):
        return len(self.content)

    def reads_from_url(self):
        self.content = ''.join(cf.utils.smartopen(self.content))
        return self

    def remove_chinese_punctuation(self):
        punctuations = set('，‘’“”。！？：（）、《》')
        self.content = self.content.replace('\n', '').replace('\r', '')
        for punctuation in punctuations:
            self.content = self.content.replace(punctuation, '')
        return self

    def remove_modal(self):
        modal = set('嗯哦呀呢噢啊唉呃吧')
        for modal_word in modal:
            self.content = self.content.replace(modal_word, '')
        return self

    def remove_english_char(self):
        self.content = re.sub(r'[a-zA-Z]', '', self.content)
        return self

    def remove_digit(self):
        self.content = re.sub(r'\d', '', self.content)
        return self

    def replace(self, old, new):
        self.content = self.content.replace(old, new)
        return self

    def split(self, sep: str) -> List[str]:
        return self.content.split(sep)

    def strip(self, __chars: str):
        self.content = self.content.strip(__chars)
        return self


class EnglishTextCleaner(object):
    def __init__(self,
                 lower: bool = False,
                 remove_stopwords: bool = False) -> None:
        self.lower = lower
        self.remove_stopwords = remove_stopwords

    def keep_alphabet(self, text: str) -> str:
        text = re.sub(
            r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?",
            " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def filter_stopwords(self, text: str) -> str:
        try:
            from nltk.corpus import stopwords
        except LookupError:
            cf.warning('nltk.corpus.stopwords not found')
            import nltk
            nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        text = ' '.join(
            [word for word in text.split() if word not in stop_words])
        return text

    def __call__(self, text: str) -> str:
        text = self.keep_alphabet(text)
        if self.lower:
            text = text.lower()
        if self.remove_stopwords:
            text = self.filter_stopwords(text)
        return text


class TrainData(object):
    """ Data holder
    """
    def __init__(self, attrs: Dict[str, Any]) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)

    def split(self,
              test_size: float = 0.2,
              random_state: int = 2023) -> 'TrainData':
        """ Assume train data 'x' and label 'y' is already assigned
        return x, xv, y, yv
        """
        return train_test_split(self.x,
                                self.y,
                                test_size=test_size,
                                random_state=random_state)


class SentenceList(object):
    """Text data wrapper
    """
    def __init__(self, sentences: List[List[str]]) -> None:
        self.sentences = sentences

    def clean(self):
        """Do text cleaning
        """
        cclist = []
        for text in self.sentences:
            text = EnglishTextCleaner(lower=True).__call__(text)
            cclist.append(text)
        self.sentences = cclist
        return self

    def tokenize(self,
                 num_words: int = 10000,
                 oov_token: str = '<OOV>') -> List[List[str]]:
        from tensorflow.keras.preprocessing.text import Tokenizer
        cf.info(f'Tokenizing texts')
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.tokenizer.fit_on_texts(self.sentences)
        self.sentences = self.tokenizer.texts_to_sequences(self.sentences)
        return self

    def pad_sequences(self,
                      padding: str = 'pre',
                      maxlen: int = 1000) -> List[List[str]]:
        cf.info(f'Padding sequences')
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        self.sentences = pad_sequences(self.sentences,
                                       maxlen=maxlen,
                                       padding=padding)
        return self

    def tfidf(self, inplace: bool = True):
        self.vectorizer = TfidfVectorizer()
        res = self.vectorizer.fit_transform(self.sentences)
        key = 'sentences' if inplace else 'sentences_tfidf'
        setattr(self, key, res)
        return self


class LabelData(object):
    def __init__(self, labels: List[int]) -> None:
        self.labels = labels

    def encode(self) -> List[int]:
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        return self

    def one_hot(self) -> 'LabelData':
        from sklearn.preprocessing import OneHotEncoder
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.labels = self.one_hot_encoder.fit_transform(self.labels)
        return self

    def to_category(self, num_classes: int = 10) -> List[str]:
        import tensorflow as tf
        self.labels = tf.keras.utils.to_categorical(self.labels,
                                                    num_classes=num_classes)
        return self


class DataManager(object):
    def split(self, X, y, test_size: float = 0.2, random_state: int = 0):
        return train_test_split(X,
                                y,
                                test_size=test_size,
                                random_state=random_state)

    def normalize(self, X):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        _ = scaler.fit_transform(X)
        return _, scaler

    def minmax(self, X):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        _ = scaler.fit_transform(X)
        return _, scaler

    def jb_cut(self, X: List, remove_punc: bool = False) -> List:
        us = [' '.join(jieba.lcut(u)) for u in X]
        if remove_punc:
            return [strip_punc(u) for u in us]
        return us

    def load_data_from_directory(self, dirname: str):
        '''Load data from a directory'''
        cf.info('loading data from dir', dirname)
        train_file = os.path.join(dirname, 'train.csv')
        test_file = os.path.join(dirname, 'test.csv')
        submission_file = os.path.join(dirname, 'submission.csv')
        submission_file_2 = os.path.join(dirname, 'sample_submission.csv')

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        submission = None
        if cf.io.exists(submission_file):
            submission = pd.read_csv(submission_file)
        elif cf.io.exists(submission_file_2):
            submission = pd.read_csv(submission_file_2)
        return df_train, df_test, submission

    def save_prediction(self,
                        y_pred,
                        target_name: str,
                        demo_file: str,
                        location: str = 'prediction.csv'):
        cf.info(
            f'save prediction to file {location}, target name: {target_name}')
        df_demo = pd.read_csv(demo_file)
        df_demo[target_name] = y_pred
        df_demo.to_csv(location, index=False)

    def pca(self, X, n_components: int = 10):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        return X, pca

    def count_vectorize(
            self, X: pd.DataFrame) -> Tuple[pd.DataFrame, CountVectorizer]:
        cv_ = CountVectorizer()
        _X = cv_.fit_transform(X)
        return _X, cv_

    def tfidf_vectorize(
            self, X: pd.DataFrame) -> Tuple[pd.DataFrame, CountVectorizer]:
        vectorizer = TfidfVectorizer()
        _X = vectorizer.fit_transform(X)
        return _X, vectorizer


def any_cn(X) -> bool:
    '''Decides any Chinese char was contained'''
    if isinstance(X, str):
        X = [X]
    return any(cf.nstr(s).is_cn() for s in X)


def datainfo(sequences: list) -> None:
    len_list = list(map(len, sequences))

    def percentile(n: int):
        return int(np.percentile(len_list, n))

    print('{:<30} {}'.format('Size of sequence:', len(sequences)))
    print('{:<30} {}'.format('Maximum length:', max(len_list)))
    print('{:<30} {}'.format('Minimum length:', min(len_list)))
    print('{:<30} {}'.format('Percentile 90 :', percentile(90)))
    print('{:<30} {}'.format('Percentile 80 :', percentile(80)))
    print('{:<30} {}'.format('Percentile 20 :', percentile(20)))
    print('{:<30} {}'.format('Percentile 10 :', percentile(10)))
    print('{:<30} {}'.format('The mode is :', stats.mode(len_list)[0]))

    if any_cn(sequences):
        import jieba
        sequences = [jieba.lcut(s) for s in sequences]
    _, tokenizer = tokenize(sequences)
    print('unique words count {}'.format(len(tokenizer.word_index)))


def list_physical_devices() -> list:
    from tensorflow.python.eager import context
    _devices = context.context().list_physical_devices()
    cf.info(_devices)
    return _devices


def tokenize(X: list, max_feature: int = 10000) -> list:
    cf.info(f'Tokenizing texts')
    from tensorflow.keras.preprocessing.text import Tokenizer
    tok = Tokenizer(num_words=max_feature)
    tok.fit_on_texts(X)
    ans = tok.texts_to_sequences(X)
    return ans, tok


def label_encode(y: list, return_processor: bool = False) -> np.ndarray:
    '''Encode labels into 0, 1, 2...'''
    cf.info(f'Getting labels. Return encoder is set to {return_processor}')
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    y_categories = enc.fit_transform(y)
    return (y_categories, enc) if return_processor else y_categories


def onehot_encode(y: list, return_processor: bool = False) -> np.ndarray:
    '''input format: y =[['red'], ['green'], ['blue']]
    '''
    cf.info(
        f'Getting one hot encode labels. Return encoder is set to {return_processor}'
    )
    assert isinstance(y[0], list) or isinstance(y[0], np.ndarray)
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y_categories = enc.fit_transform(y)
    return (y_categories, enc) if return_processor else y_categories


def pad_sequences(sequences, **kwargs):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = pad_sequences(sequences, **kwargs)
    return seq


def birdview(df: pd.DataFrame) -> None:
    print('{:<20}:{}'.format(cf.fp.cyan('shape'), df.shape))
    print('{:<20}:{}'.format(cf.fp.cyan('columns'), df.columns.tolist()))
    print('{:<20}:\n{}'.format(cf.fp.cyan('head'), df.head()))
    print('{:<20}:\n{}'.format(cf.fp.cyan('info'), df.info()))
    print('{:<20}:\n{}'.format(cf.fp.cyan('describe'), df.describe()))


once = DataManager()

tools = Namespace(birdview=birdview,
                  train_test_split=train_test_split,
                  split=train_test_split)

import warnings


class Stopwords(object):
    def __init__(self):
        warnings.warn(
            "premium.data.Stopwords() is deprecated; use premium.corpus.stopwords instead.")
        pass

    def _load_stopwords_file(self, file_name: str) -> set:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(current_dir)
        stopwords_file = os.path.join(parent_dir, f'localdata/{file_name}')
        return set(cf.io.read(stopwords_file))

    @property
    def cn(self) -> set:
        return self._load_stopwords_file('cn_stopwords.txt')

    @property
    def en(self) -> set:
        return self._load_stopwords_file('en_stopwords.txt')


data = Namespace(contraction_dict=contraction_dict)