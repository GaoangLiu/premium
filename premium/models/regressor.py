#!/usr/bin/env python
import abc

import codefast as cf
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from codefast.logger import test
from optuna.integration import XGBoostPruningCallback
from sklearn.ensemble import VotingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import premium as pm
from premium.preprocess import any_cn, jb_cut

from .base import BaseModel


class BaseRegressor(metaclass=abc.ABCMeta):
    def __init__(self, X, y, test_size: float = 0.2):
        self.X = jb_cut(X) if any_cn(X) else X
        self.y = y
        self.test_size = test_size
        self.model_type = 'basemodel'
        self.model = None
        self.stop_words = []
        self.cv = None
        self.scoring = None
        self.n_iter = 100


class LR(BaseRegressor):
    def __init__(self, X, y, test_size: float = 0.2):
        super(LR, self).__init__(X, y, test_size)
        self.parameters = {}

    def build_model(self):
        self.model = LinearRegression()


class XgboostRegressor(BaseModel):
    def __init__(self, X, y, X_test, test_size: float = 0.2):
        super(XgboostRegressor, self).__init__(X, y, X_test)

    def objective(self, trial):
        param = {
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0), \
            "gamma": trial.suggest_loguniform('gamma', 1e-4, 1e4), \
            'eta': trial.suggest_float('eta', 0.007, 0.013),\
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4), # L2 regularization \
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4),\
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-4, 1e4), \
            "subsample": trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1), \
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05), \
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.2, 0.9, 0.1), \
            'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.2, 0.9, 0.1),\
            'n_estimators': 1000,
            'max_depth': trial.suggest_int('max_depth', 3, 30),\
            'n_jobs': self.extra_parameters.get('n_jobs', 4),
        }
        model = XGBRegressor(**param)
        # pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
        n_splits = self.extra_parameters.get('n_splits', 3)
        n_repeats = self.extra_parameters.get('n_repeats', 2)
        random_state = self.extra_parameters.get('random_state', 28947)

        rkf = RepeatedKFold(n_splits=n_splits,
                            n_repeats=n_repeats,
                            random_state=random_state)
        scores = cross_val_score(model,
                                 self.X,
                                 self.y,
                                 cv=rkf,
                                 scoring='neg_root_mean_squared_error')
        return -1 * np.mean(scores)

    def build_model(self, input_params: dict = {}):
        _params = {
            'max_depth': 3,
            'min_child_weight': 5,
            'n_estimators': 3000,
            'learning_rate': 0.008,
            'subsample': 0.4,
            'booster': 'gbtree',
            'colsample_bytree': 0.6,
            'reg_lambda': 5,
            'reg_alpha': 5,
            'reg_alpha': 32,
            'n_jobs': 13,
            'alpha': 0.5,
            'random_state': 123
        }
        _params.update(input_params)
        self.model = XGBRegressor(**_params)
        return self.model


class MlpRegressor(BaseModel):
    def __init__(self, X, y, test_size: float = 0.2):
        super(MlpRegressor, self).__init__(X, y, test_size)

    def build_model(self):
        self.model = MLPRegressor(hidden_layer_sizes=100,
                                  early_stopping=True,
                                  n_iter_no_change=100,
                                  solver='adam',
                                  shuffle=True,
                                  random_state=42)


class CatboostRegressor(BaseModel):
    def __init__(self,
                 X,
                 y,
                 X_test,
                 iterations: int = 1000,
                 n_splits: int = 10,
                 task_type='CPU',
                 eval_metric: str = 'RMSE',
                 seed: int = 63):
        super(CatboostRegressor, self).__init__(X, y, X_test)
        self.iterations = iterations
        self.n_splits = n_splits
        self.task_type = task_type  # use GPU or CPU
        self.eval_metric = eval_metric
        self.seed = seed

    def objective(self, trial):
        data, target = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(data,
                                                            target,
                                                            test_size=0.25,
                                                            random_state=42)
        params = {
            'iterations': trial.suggest_int("iterations", 500, 3000), \
            'od_wait': trial.suggest_int('od_wait', 200, 2000), \
            'loss_function': 'RMSE', \
            'eval_metric': 'RMSE', \
            'leaf_estimation_method': 'Newton', \
            'bootstrap_type': 'Bernoulli', \
            'learning_rate': trial.suggest_uniform('learning_rate', 0.02, 1), \
            'reg_lambda': trial.suggest_uniform('reg_lambda', 1e-5, 100),\
            'subsample': trial.suggest_uniform('subsample', 0, 1), \
            'random_strength': trial.suggest_uniform('random_strength', 10, 50), \
            'depth': trial.suggest_int('depth', 1, 15), \
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30), \
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),\
            'task_type': 'CPU',
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds=500,
                  verbose=False)

        y_preds = model.predict(X_test)
        loss = pm.libra.rmse(y_test, y_preds)

        return loss

    def build_model(self, input_params: dict = {}):
        _params = {
            'depth': 6,
            'iterations': 1000,
            'learning_rate': 0.024,
            'l2_leaf_reg': 20,
            'random_strength': 1.5,
            'grow_policy': 'Depthwise',
            'leaf_estimation_method': 'Newton',
            'bootstrap_type': 'Bernoulli',
            'thread_count': 4,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'od_type': 'Iter',
            'task_type': self.task_type,
            'early_stopping_rounds': 500,
            'verbose': 100,
            'random_state': 63
        }
        _params.update(input_params)
        cf.info('catboost build model params', _params)
        # only one of the parameters l2_leaf_reg, reg_lambda should be initialized.
        if 'reg_lambda' in _params and 'l2_leaf_reg' in _params:
            _params.pop('l2_leaf_reg', None)

        # only one of the parameters od_wait, early_stopping_rounds should be initialized.
        if 'od_wait' in _params:
            del _params['early_stopping_rounds']

        self.model = CatBoostRegressor(**_params)
        cf.info('catboost model created.')
        return self.model

