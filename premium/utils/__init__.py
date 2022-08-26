import codefast as cf
import hashlib
from typing import List,Union,Dict,Tuple
import numpy as np

def md5sum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
    
    
def auto_set_label_num(y: List[Union[str,
                                     int]]) -> Tuple[Dict, List[int]]:
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
    cf.info('Export {label, id} map to /tmp/label_map.json')
    cf.js.write(label_map, '/tmp/label_map.json')
    return label_map, new_y

    