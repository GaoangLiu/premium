#!/usr/bin/env python
from typing import List

import codefast as cf

from .utils import LocalData

class BookInfo(object):
    def __init__(self, name:str, location:str) -> None:
        self.name=name
        self.location=location

    def load(self)->str:
        if self.location.startswith('http'):
            return cf.net.get(self.location)
        elif cf.io.exists(self.location):
            return cf.io.reads(self.location)
        else:
            raise ValueError(f'{self.location} not found')


class BookReader(object):
    def __init__(self, book_list:List[BookInfo]):
        self.book_list=book_list

    def read(self, book_name:str)->str:
        b = [book for book in self.book_list if book.name==book_name]
        return b[0].load()

    def list(self)->List[str]:
        return [book.name for book in self.book_list]


bookreader = BookReader([BookInfo('soledad', LocalData('soledad.txt').fullpath())])

