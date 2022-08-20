#!/usr/bin/env python
import abc

import codefast as cf
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from codefast.logger import test
from optuna import TPESampler
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import (RepeatedKFold, cross_val_score,
                                     train_test_split)
from xgboost import XGBClassifier, XGBRegressor

import premium as pm
from premium.preprocess import any_cn, jb_cut


class XgboostClassifier(XGBClassifier):
    def __init__(self, *, X=None, y=None, **kwargs):
        super().__init__(*kwargs)
        self.X = X
        self.y = y

    def optuna(self) -> float:
        def _objective(trial: optuna.Trial, X, y) -> float:
            param = {
                    "n_estimators" : trial.suggest_int('n_estimators', 100, 10000), \
                    'max_depth':trial.suggest_int('max_depth', 2, 25), \
                    'reg_alpha':trial.suggest_int('reg_alpha', 0, 5), \
                    'reg_lambda':trial.suggest_int('reg_lambda', 0, 5), \
                    'min_child_weight':trial.suggest_int('min_child_weight', 0, 5), \
                    'gamma':trial.suggest_int('gamma', 0, 5), \
                    'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5), \
                    'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01), \
                    'nthread' : -1
                }
            model = XGBClassifier(**param)

            return cross_val_score(model, X, y, cv=3).mean()

        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(lambda trial: _objective(trial, self.X, self.y),
                       n_trials=50)
