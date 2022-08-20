#!/usr/bin/env python
import random
from collections import defaultdict
import codefast as cf
from typing import List, Dict, Tuple, Set, Optional, Union
import csv


class _Row(object):
    """ A row of data. [keys] are required to be in the order of the csv file.
    """

    def __init__(self, row: list, keys: list) -> None:
        for i, v in enumerate(row):
            setattr(self, keys[i], v)

    def __repr__(self) -> str:
        return ', '.join([str(getattr(self, k)) for k in self.__dict__.keys()])

    def to_list(self) -> list:
        return [getattr(self, k) for k in self.__dict__.keys()]

    def len(self) -> int:
        return len(self.to_list())


class CsvReader(object):
    def __init__(self, csv_file: str) -> None:
        self.column_attributes = {'max_length': defaultdict(int)}
        self.list = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            self.headers = next(reader)
            self.str_headers = ', '.join(self.headers)
            self.columns = self.headers
            for e in reader:
                self.list.append(_Row(e, self.headers))
                for i, k in enumerate(self.headers):  # update column attributes
                    if len(e[i]) > self.column_attributes['max_length'][k]:
                        self.column_attributes['max_length'][k] = len(e[i])

    def shape(self):
        return len(self.list), len(self.list[0].__dict__.keys())

    def to_csv(self, csv_file: str):
        """write data to csv file"""
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            for e in self.list:
                writer.writerow(e.to_list())

    def __setitem__(self, key: str, value: Union[str, int, List]):
        """ Add a new column to the csv file or overwrite an existing column,
            value can be a list or a constant value.
        """
        if key not in self.columns:
            self.columns.append(key)
            self.str_headers = ', '.join(self.columns)  # update headers
        if isinstance(value, list):
            assert len(value) == len(
                self.list), "length of value list must be equal to the length of the csv file"
            for i, e in enumerate(self.list):
                setattr(e, key, value[i])
        elif isinstance(value, str):
            for e in self.list:
                setattr(e, key, value)
        else:
            raise ValueError('value must be a list or a constant value')

    def __getitem__(self, key: str) -> List:
        idx = self.columns.index(key)
        return [e[idx] for e in self.list]

    def __print_sublist(self, sublist: list, format: bool) -> 'CsvReader':
        if not format:
            for e in sublist:
                print(e)
        else:
            _max_lengthes = [0] * sublist[0].len()
            for sl in sublist:
                for i, e in enumerate(sl.to_list()):
                    if len(str(e)) > _max_lengthes[i]:
                        _max_lengthes[i] = len(str(e))
            _str = ' '.join(['{:<' + str(l) + '}' for l in _max_lengthes])
            for sl in sublist:
                print(_str.format(*sl.to_list()))
        return self

    def head(self, n=10, format: bool = False) -> 'CsvReader':
        """ Print the first n rows of the csv file."""
        print(self.str_headers)
        self.__print_sublist(self.list[:n], format)
        return self

    def sample(self, n: int = 10, format: bool = False) -> 'CsvReader':
        """ Return a random sample of n rows of the csv file. """
        random.shuffle(self.list)
        return self.head(n, format=format)
