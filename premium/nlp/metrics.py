import numpy
from argparse import Namespace
from typing import Dict, List, Tuple


def diarization_error_rate(reference: Dict[float, int],
                           hypothesis: Dict[float, int]) -> float:
    """
    Diarization Error Rate. Reference: 
    Reference: 
        1. https://digitalassets.lib.berkeley.edu/etd/ucb/text/Knox_berkeley_0028E_13539.pdf
        2. https://is.gd/44YNEN
    Args:

        reference: real speech diarization result
        hypothesis: predicted speech diarization result
        Each key is a time point accurate to 0.01s
        , and each value is a speaker ID. More specifically, 
            1 means the first speaker, 2 means the second speaker, and so on. 
            0 is reserved for non-speech.
    Returns:
        float: error rate
    """
    total_len = len([k for k, v in reference.items() if v != 0])
    confusion_error, miss_error, false_alarm_error = 0, 0, 0

    for k, v in reference.items():
        if v != 0:
            if k not in hypothesis:
                miss_error += 1
            elif v != hypothesis[k]:
                confusion_error += 1
        else:
            if k in hypothesis and hypothesis[k] != 0:
                false_alarm_error += 1

    return (confusion_error + miss_error + false_alarm_error) / total_len


def cer(r: str, h: str):
    '''Character Error Rate
    (S + D + I) /N 
    S: substitution
    D: Deletion
    I: Insertion
    Args: 
        r: reference
        h: hypothesis
    '''
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r))


def word_correct_rate(r: str, h: str):
    '''word correct rate
    1 - (S + D) /N, i.e., not counting insertions
    S: substitution
    D: Deletion
    Args: 
        r: reference
        h: hypothesis
    '''
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, deletion)
    return d[len(r)][len(h)] / float(len(r))


nlp_metrics = Namespace(
    diarization_error_rate=diarization_error_rate,
    der=diarization_error_rate,
    cer=cer,
    wcr=word_correct_rate
)
