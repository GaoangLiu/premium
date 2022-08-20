from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import codefast as cf
from premium.measure import libra
import datetime

import pandas as pd
import random
import numpy as np


class Classifier(object):
    def __init__(self, clf: Any) -> None:
        self.clf = clf

    def cv(self,
           X,
           y,
           cv=3,
           persist_result: bool = True,
           scoring: Optional[str] = 'accuracy'):
        kfold = StratifiedKFold(n_splits=cv, shuffle=True)
        scores = cross_val_score(estimator=self.clf,
                                 X=X,
                                 y=y,
                                 cv=kfold,
                                 n_jobs=-1,
                                 scoring=scoring)
        result = {
            'cv_scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std(),
            'name': str(self.clf.__class__.__name__)
        }
        op = '/tmp/{}_{}.json'.format(
            self.clf.__class__.__name__,
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')).lower()
        if persist_result:
            cf.js.write(result, op)
        cf.info('cv scores: {}'.format(scores))
        cf.info('average: {}'.format(scores.mean()))
        cf.info('std: {}'.format(scores.std()))
        cf.info('exported to {}'.format(op))
        return result

    def cv2(self, train, labels, cv=3, test=None):
        """ 
        Args:
        """
        scores = []
        y_preds = []
        for i in range(cv):
            start_time = datetime.datetime.now()
            cf.info('cross validation round {} started'.format(i))
            x, xv, y, yv = train_test_split(train,
                                            labels,
                                            test_size=0.2,
                                            random_state=random.randint(
                                                0, 1 << 20))
            self.clf.fit(train, labels)
            preds = self.clf.predict(xv)
            scores.append(libra.accuracy_score(yv, preds))
            if test is not None:
                y_preds.append(self.clf.predict(test))
            end_time = datetime.datetime.now()
            cf.info(
                'cross validation round {} finished in {} seconds, score: {}'.
                format(i, (end_time - start_time).seconds, scores[-1]))
        for i, _score in enumerate(scores):
            cf.info('cv round {} score: {}'.format(i, _score))
        cf.info('average score: {}'.format(np.mean(scores)))
        cf.info('std: {}'.format(np.std(scores)))
        if y_preds:
            from premium.data.postprocess import mode
            y_preds = mode(np.array(y_preds), 'col').flatten()
        return scores, y_preds

    def fit(self, X, y, *args, **kwargs):
        self.clf.fit(X, y, *args, **kwargs)
        cf.info('fit completes')
        return self

    def score(self, Xt, yt) -> float:
        """ Every estimator or model in Scikit-learn has a score method after being trained on the data.
        For models outside of Scikit-learn or being customized, this method has to added manually.
        """
        try:
            return self.clf.score(Xt, yt)
        except Exception as e:
            cf.warning(e)
            preds = self.clf.predict(Xt)
            return libra.accuracy_score(yt, preds)

    def predict_proba(self, Xt):
        return self.clf.predict_proba(Xt)

    def test(self, Xt, yt, test_metric: Callable = libra.metrics):
        cf.info('Test on test set with metric: {}'.format(
            test_metric.__name__))
        y_pred = self.clf.predict(Xt)
        self.scores = test_metric(yt, y_pred)
        cf.info('Test score(s): {}'.format(self.scores))
        return self

    def predict(self, Xt):
        return self.clf.predict(Xt)
