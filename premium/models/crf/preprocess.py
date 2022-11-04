#!/usr/bin/env python
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union, NewType

import codefast as cf
# parser = argparse.ArgumentParser()
# parser.add_argument("corpus_file", help="corpus file for training input")
# args = parser.parse_args()


def get_label(text: str) -> List[str]:
    """[4000来块]钱吧。 -> [4/M, 0/M, 0/M, 0/M, 来/M, 块/M, 钱/O, 吧/O]
    """
    is_m = False
    x, y = [], []
    for t in list(text):
        if t == '[':
            is_m = True
        elif t == ']':
            is_m = False
        else:
            x.append(t)
            y.append('M' if is_m else 'O')
            # res.append(t + '/M' if is_m else t + '/O')
    return x, y


def get_label_sbme(text: str) -> Tuple[List]:
    """[4000来块]钱吧。[万]。 -> [[4, 0, 0, 0, 来, 块, 钱, 吧, 万], [B, M, M, M, M, E, O, O, O, S]]
    """
    s, b = False, False
    lst = []
    chars = list(text)
    for i, t in enumerate(chars):
        if t == '[':
            s = True
            b = True
        elif t == ']':
            s = False
            b = False
        else:
            x, y = t, None
            if s == False and b == False:
                lst.append((x, 'O'))
                continue
            if chars[i-1] == '[':
                if chars[i+1] == ']':
                    y = 'S'
                else:
                    y = 'B'
            else:
                if chars[i+1] == ']':
                    y = 'E'
                else:
                    y = 'M'
            lst.append((x, y))
    return lst


Char = NewType('Char', str)


def restore_from_label(token_list: List[Char], label_list: List[Char]) -> str:
    """ restore a sentence from label
    Args:
        token_list: a list of tokens, e.g., ['我', '吃', '西', '瓜']
        label_list: a list of labels, e.g., ['O', 'O', 'B', 'E']
    Returns:
        a sentence, e.g., '我吃[西瓜]'
    """
    res = []
    for t, l in zip(token_list, label_list):
        if l == 'O':
            res.append(t)
        elif l == 'B':
            res.append('[' + t)
        elif l == 'E':
            res.append(t + ']')
        elif l == 'S':
            res.append('[' + t + ']')
        elif l == 'M':
            res.append(t)
    return ''.join(res)
