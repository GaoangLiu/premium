import hashlib
import os
from typing import Dict, List, Tuple, Union

import codefast as cf
import numpy as np


def cf_unless_tmp(file_name: str):
    tmp_file = '/tmp/' + file_name
    if not os.path.exists(tmp_file):
        cf.net.download('https://host.ddot.cc/{}'.format(file_name), tmp_file)
    return tmp_file


def try_import(libname: str):
    try:
        import importlib
        importlib.import_module(libname)
    except ImportError:
        import os
        print('{} not installed, please install gensim first'.format(libname))
        os.system('pip install {}'.format(libname))


def md5sum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def auto_set_label_num(y: List[Union[str, int]],
                       working_dir: str = '/tmp/') -> Tuple[Dict, List[int]]:
    """ Automatically set the number of labels based on the labels.
    If it is binary classification, then new label is like [0, 1, 1, 0], 
    if it is multi-classification, then new label is like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    Return: 
        - A map from old label to new label
        - A list of new label
    """
    unique_labels = set(y)
    label_map = dict(zip(unique_labels, range(len(unique_labels))))
    new_y = np.array([label_map[yi] for yi in y])
    cf.info('Export [label, id] map to {}'.format(
        os.path.join(working_dir, 'label_map.json')))
    cf.js.write(label_map,
                '{}'.format(os.path.join(working_dir, 'label_map.json')))
    cf.info(label_map)
    return label_map, new_y
