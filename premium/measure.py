#!/usr/bin/env python
from typing import Dict

import codefast as cf
import numpy
from codefast.utils import deprecated
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)

red = cf.fp.red
green = cf.fp.green


class Libra(object):
    '''Metrics'''
    def rmse(self, y_true, y_pred) -> float:
        return mean_squared_error(y_true, y_pred, squared=False)

    def roc(self, y_true, y_pred) -> float:
        return roc_auc_score(y_true, y_pred)

    def auc(self, y_true, y_pred) -> float:
        return auc(y_true, y_pred)

    def mse(self, y_true, y_pred) -> float:
        return mean_squared_error(y_true, y_pred)

    def f1_score(self, y_true, y_pred, average: str = 'binary') -> float:
        return f1_score(y_true, y_pred, average)

    def accuracy_score(self, y_true, y_pred) -> float:
        return accuracy_score(y_true, y_pred)

    def recall_score(self, y_true, y_pred) -> float:
        average = 'micro' if len(set(y_true)) == 2 else 'macro'
        return recall_score(y_true, y_pred, average=average)

    def precision_score(self, y_true, y_pred) -> float:
        average = 'micro' if len(set(y_true)) == 2 else 'macro'
        return precision_score(y_true, y_pred, average=average)

    def confusion_matrix(self, y_true, y_pred) -> float:
        return confusion_matrix(y_true, y_pred)

    def f1_score_manual(self, acc: float, rec: float) -> float:
        return 2 * (acc * rec) / (acc + rec)

    def metrics(self,
                y_true,
                y_pred,
                f1_score_average: str = 'average') -> Dict:
        acc = self.accuracy_score(y_true, y_pred)
        rec = self.recall_score(y_true, y_pred)
        scores = {
            'accuracy': acc,
            'precision': rec,
            'f1_score': self.f1_score(y_true, y_pred, f1_score_average),
            'fi_score_manual': self.f1_score_manual(acc, rec),
            'recall': self.recall_score(y_true, y_pred),
            'confusion_matrix': self.confusion_matrix(y_true, y_pred)
        }
        for k, v in scores.items():
            v = '\n{}'.format(v) if isinstance(v, numpy.ndarray) else v
            print('{:<20}: {}'.format(k, v))
        return scores


libra = Libra()


@deprecated(cf.fp.red('Use pm.libra.metrics(y_true, y_pred) instead.'))
def metrics(y_true, y_pred):
    print('{:<20}: {}'.format('Accuracy score', accuracy_score(y_true, y_pred)))
    print('{:<20}: {}'.format('Recall score', recall_score(y_true, y_pred)))
    print('{:<20}: \n{}'.format('Confusion matrix',
                                confusion_matrix(y_true, y_pred)))
    print('{:<20}: {}'.format('f1_score', f1_score(y_true, y_pred)))
