#!/usr/bin/env python
from typing import List

import codefast as cf

from .utils import LocalData


class stopwords(object):
    def __init__(self) -> None:
        self._cn_stopwords = None
        self._en_stopwords = None

    @property
    def cn_stopwords(self) -> List[str]:
        if self._cn_stopwords is None:
            file_path = LocalData('cn_stopwords.txt').fullpath()
            self._cn_stopwords = cf.io.read(file_path)
        return self._cn_stopwords

    @property
    def en_stopwords(self) -> List[str]:
        if self._en_stopwords is None:
            file_path = LocalData('en_stopwords.txt').fullpath()
            self._en_stopwords = cf.io.read(file_path)
        return self._en_stopwords

    @staticmethod
    def words(lang: str = 'cn') -> List[str]:
        """Get stopwords list
        """
        if lang == "cn" or lang == "chinese":
            return stopwords().cn_stopwords
        elif lang == "en" or lang == "english":
            return stopwords().en_stopwords
        else:
            raise ValueError('lang must be cn or en')
