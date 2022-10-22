#!/usr/bin/env python
import pickle
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import codefast as cf
import sklearn_crfsuite


class CRF(object):
    """ A skleanr crfsuite wrapper for fast model training and deployment
    """

    def __init__(self, X: List, y: List, feature_template: str = None) -> None:
        """
        Args:
            X: input data
            y: labels
            feature_template: path of feature template file, follow https://taku910.github.io/crfpp/ for template example
            A feature templte demo:
            # Unigram
            U00:%x[-2,0]
            U01:%x[-1,0]
            U05:%x[-1,0]/%x[0,0]
            U06:%x[0,0]/%x[1,0]

            U10:%x[-2,1]
            U15:%x[-2,1]/%x[-1,1]

            U20:%x[-2,1]/%x[-1,1]/%x[0,1]
            U21:%x[-1,1]/%x[0,1]/%x[1,1]

            # Bigram
            B
        """
        self.X = X
        self.y = y
        self.feature_template = feature_template
        self.is_feature_extracted = False
        self.model = None

    def _word2features(self, s: List[Tuple],
                       i: int) -> Dict[str, Union[str, List[str]]]:
        """ Convert a word to features. 
        Usually, we don't have much information on samples, so we only use words
        as features. But if we do have extra information, such as POS tag, NER tag, etc,
        then we can leverage them to improve the performance by adding features
        such as 'U[-1]_POS': s[i-1][1], which means the POS tag of the previous word.
        """
        return {
            'U[0]':
            s[i][0],
            'U[-1]':
            s[i - 1][0] if i > 0 else '',
            'U[-2]':
            s[i - 2][0] if i > 1 else '',
            'U[+1]':
            s[i + 1][0] if i < len(s) - 1 else '',
            # 'U[+2]':
            # s[i + 2][0] if i < len(s) - 2 else '',
            'B[-1]': [s[i - 1][0], s[i][0]] if i > 0 else '',
            'B[+1]': [s[i][0], s[i + 1][0]] if i < len(s) - 1 else ''
            # 'B[-1]B[1]': [s[i - 1][0], s[i][0], s[i + 1][0]]
            # if i > 0 and i < len(s) - 1 else '',
        }

    def _sent2features(self, s: List):
        """ Convert a sentence to features
        """
        return [self._word2features(s, i) for i, _ in enumerate(s)]

    def extract_features(self):
        """ Extract features from input data
        """
        self.X = [self._sent2features(s) for s in self.X]
        self.is_feature_extracted = True
        cf.info('features extracted')
        return self.X

    def fit(self):
        if not self.is_feature_extracted:
            self.X = self.extract_features()
        self.model = sklearn_crfsuite.CRF(algorithm='lbfgs',
                                          c1=0.1,
                                          c2=0.1,
                                          max_iterations=300,
                                          verbose=True,
                                          all_possible_transitions=True)
        cf.info('crf model created')
        self.model.fit(self.X, self.y)

    @classmethod
    def load_model(cls, model_path: str) -> 'CRF':
        return pickle.load(open(model_path, 'rb'))

    def save_model(self, model_path: str):
        pickle.dump(self, open(model_path, 'wb'))
        return True

    def predict(self, X: List):
        """ Predict labels for input data
        """
        return self.model.predict(X)

    def evaluate(self):
        pass

    def predict_proba(self):
        pass
