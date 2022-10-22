#!/usr/bin/env python
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

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
