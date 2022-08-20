#!/usr/bin/env python
import abc
from io import UnsupportedOperation
from math import gamma

import codefast as cf
import joblib
import numpy as np
import optuna
from genericpath import samefile
from numpy.lib.function_base import average
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

import premium as pm
from premium.preprocess import any_cn


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, X, y, X_test):
        self.X = pm.once.jb_cut(X) if any_cn(X) else X
        self.y = y
        self.X_test = X_test
        self.X_train = None
        self.y_train = None
        self.test_size = 0.2
        self.model_type = 'basemodel'
        self.stop_words = []
        self.countervectorize = False
        self.cv = None
        self.save_model = False
        self.metrics = pm.libra.rmse
        self.n_trials = 10
        self.extra_parameters = {}  # For other more parameters
        self.n_splits = 10  # KFold splits number
        self.seed = 63

    def preprocess(self):
        cf.info('stop word is set to ', repr(self.stop_words))
        X_train, X_val, y_train, y_val, idx1, idx2 = train_test_split(
            self.X,
            self.y,
            np.arange(len(self.X)),
            random_state=self.seed,
            test_size=self.test_size)

        if self.countervectorize:
            cv = CountVectorizer(min_df=1,
                                 max_df=1.0,
                                 token_pattern='\\b\\w+\\b',
                                 stop_words=self.stop_words)
            X_train = cv.fit_transform(X_train)
            X_val = cv.transform(X_val)
            self.cv = cv

            if self.save_model:
                f_vocabulary = cf.io.tmpfile('cv', 'json')
                cf.js.write(cv.vocabulary_, f_vocabulary)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.indices = [idx1, idx2]
        self.indices = {'train': idx1, 'val': idx2}
        cf.info('Preprocess completed.')
        return X_train, X_val, y_train, y_val, idx1, idx2

    def postprocess_prediction(self, y_pred: list) -> list:
        # Post process prediction, e.g., regression result to classification result
        return y_pred

    def build_model(self):
        cf.info('build model completed.')

    def k_fold(self,
               n_splits: int = 10,
               random_state: int = 2021,
               metric=pm.libra.rmse,
               lightgbm_eval_metric: str = 'auc'):
        '''Do K fold cross valid train and make predictions on X_test.
        '''
        cf.info('input X shape', self.X.shape)
        cf.info('input y shape', self.y.shape)
        self.kf = KFold(n_splits=n_splits,
                        shuffle=True,
                        random_state=random_state)
        y_pred = np.zeros(self.X_test.shape[0])
        oof = np.zeros(self.X.shape[0])
        average_score = 0

        for fold, (trn_idx,
                   val_idx) in enumerate(self.kf.split(self.X, self.y)):
            cf.info(f'........... FOLD {fold} ............')
            X_train, y_train = self.X.iloc[trn_idx], self.y.iloc[trn_idx]
            X_valid, y_valid = self.X.iloc[val_idx], self.y.iloc[val_idx]

            _model_type = str(type(self.model))
            cf.info('modle type', _model_type)
            if 'catboost' in _model_type:
                self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

            elif 'xgboost' in _model_type:
                self.model.fit(X_train, y_train, early_stopping_rounds=100)

            elif 'lightgbm' in _model_type:
                self.model.fit(X_train,
                               y_train,
                               eval_set=[(X_valid, y_valid)],
                               eval_metric=lightgbm_eval_metric,
                               early_stopping_rounds=200,
                               verbose=1000)
            else:
                raise Exception(f'unsupported model {_model_type}.')

            oof[val_idx] = self.model.predict(X_valid)
            y_pred += self.model.predict(self.X_test) / self.n_splits
            score = metric(y_valid, oof[val_idx])
            average_score += score / n_splits
            cf.info(f"fold {fold} - score: {score:.6f}")
        cf.info('Average score', average_score)

        return y_pred, average_score

    def fit(self):
        if self.X_train is None:
            self.preprocess()

        cf.info('start training with model', self.model)
        self.model.fit(self.X_train, self.y_train)
        cf.info(f'Training completed')

        if self.save_model:
            model_name = cf.io.tmpfile(self.model_type, 'joblib')
            joblib.dump(self.model, model_name, compress=9)
            cf.info('model saved to {}'.format(model_name))

        y_pred = self.model.predict(self.X_val)
        y_pred = self.postprocess_prediction(y_pred)
        _score = self.metrics(self.y_val, y_pred)
        cf.info('Score:', _score)
        return self.X_train, self.X_val, self.y_train, self.y_val, y_pred

    def benchmark(self,
                  n_trials: int = 10,
                  n_splits: int = 10,
                  random_state: int = 2021):
        cf.info(
            f'Do benchmark with params n_trail {n_trials}, n_splits {n_splits}, random_state {random_state}'
        )
        _best_params = self.optuna(self.objective, n_trials)
        self.build_model(_best_params)
        y_pred, average_score = self.k_fold(n_splits, random_state)
        return y_pred, average_score

    def predict(self, X_test: list) -> list:
        y_pred = self.model.predict(X_test)
        cf.info('prediction completes')
        return y_pred

    def ensure_process(self):
        if not self.X_train:
            self.preprocess()

    def optuna(self,
               objective,
               n_trials,
               _direction: str = 'minimize',
               study_name: str = 'optuna'):
        study = optuna.create_study(direction=_direction,
                                    study_name=study_name,
                                    pruner=SuccessiveHalvingPruner())

        study.optimize(objective, n_trials)

        best_trial = study.best_trial
        cf.info("Number of finished trials: {}".format(len(study.trials)))
        cf.info("Best trial: {}".format(best_trial.value))
        cf.info(best_trial.params)
        return best_trial.params
