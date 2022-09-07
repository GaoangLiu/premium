#!/usr/bin/env python
import random
import enum
import re, os, sys, joblib
from collections import defaultdict
from functools import reduce
import codefast as cf
from typing import Dict, List, Optional, Set, Tuple, Callable


class EntityTuple(object):
    """ EntityTuple(text=, label=, start=, end=) 
    """
    def __init__(self, text: str, label: str, start: int, end: int):
        self.text = text
        self.label = label
        self.start = start
        self.end = end

    def __repr__(self):
        return f'EntityTuple(text={self.text}, label={self.label}, start={self.start}, end={self.end})'

    def __eq__(self, other):
        return self.text == other.text and self.label == other.label and self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.text, self.label, self.start, self.end))

    def __len__(self):
        return len(self.text)


class EntityType(enum.Enum):
    PERSON = 'PERSON'
    ORG = 'ORG'
    GPE = 'GPE'
    LOC = 'LOC'
    FAC = 'FAC'
    PRODUCT = 'PRODUCT'
    EVENT = 'EVENT'
    WORK_OF_ART = 'WORK_OF_ART'
    LANGUAGE = 'LANGUAGE'
    DATE = 'DATE'
    TIME = 'TIME'
    PERCENT = 'PERCENT'
    MONEY = 'MONEY'
    QUANTITY = 'QUANTITY'
    ORDINAL = 'ORDINAL'
    CARDINAL = 'CARDINAL'


class EntityRecognizer(object):
    def __init__(self):
        ...

    def __call__(self, text: str) -> List[EntityTuple]:
        ...

    def __repr__(self):
        ...

    def __len__(self):
        ...