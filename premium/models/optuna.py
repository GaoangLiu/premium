#!/usr/bin/env python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import codefast as cf
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from premium.data.preprocess import TrainData
from lightgbm import LGBMClassifier


class HyperOpt(object):
    def __init__(self, td: TrainData):
        self.td = td

    def cv_objective(self, trial, _x, _y):
        model = self.create_model(trial)
        return cross_val_score(model,
                               _x,
                               _y,
                               cv=5,
                               n_jobs=-1,
                               scoring='accuracy').mean()

    def __call__(self,
                 n_trials: int = 100,
                 study_name: str = 'params-study') -> 'Trial':
        def func(trial):
            return self.cv_objective(trial, self.td.x, self.td.y)

        study = optuna.create_study(direction='maximize',
                                    study_name=study_name)
        study.optimize(func, n_trials=n_trials, catch=(Exception, ))
        trial = study.best_trial
        return trial


class SvcHyperOpt(HyperOpt):
    def __init__(self, td: TrainData) -> None:
        self.td = td

    def create_model(self, trial) -> 'SVC':
        kernel = trial.suggest_categorical(
            'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        regularization = trial.suggest_uniform('svm-regularization', 0.01, 10)
        degree = trial.suggest_discrete_uniform('degree', 1, 5, 1)
        gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
        return SVC(kernel=kernel, C=regularization, degree=degree, gamma=gamma)


class LightGBMHyperOpt(HyperOpt):
    def __init__(self, td: TrainData) -> None:
        self.td = td

    def create_model(self, trial) -> LGBMClassifier:
        param_grid = {
            # "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "n_estimators":
            trial.suggest_categorical("n_estimators", [10000]),
            "learning_rate":
            trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves":
            trial.suggest_int("num_leaves", 20, 3000, step=20),
            "max_depth":
            trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf":
            trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "lambda_l1":
            trial.suggest_int("lambda_l1", 0, 99, step=3),
            "lambda_l2":
            trial.suggest_int("lambda_l2", 0, 99, step=3),
            "min_gain_to_split":
            trial.suggest_float("min_gain_to_split", 0, 15),
            # "bagging_fraction": trial.suggest_float(
            #     "bagging_fraction", 0.2, 0.95, step=0.1
            # ),
            # "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            # "feature_fraction": trial.suggest_float(
            #     "feature_fraction", 0.2, 0.95, step=0.1
            # ),
        }
        return LGBMClassifier(objective='multiclass',
                              metrics='multi_logloss',
                              **param_grid)


class ExtraTreesHyperOpt(HyperOpt):
    def create_model(self, trial) -> ExtraTreesClassifier:
        param_grid = {
            "n_estimators":
            trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth":
            trial.suggest_int("max_depth", 3, 12),
            "min_samples_split":
            trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":
            trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":
            trial.suggest_categorical("max_features",
                                      ['auto', 'sqrt', 'log2']),
            "bootstrap":
            trial.suggest_categorical("bootstrap", [True, False]),
            "criterion":
            trial.suggest_categorical("criterion", ['gini', 'entropy']),
        }
        return ExtraTreesClassifier(**param_grid)


class RandomForestHyperOpt(HyperOpt):
    def create_model(self, trial) -> RandomForestClassifier:
        param_grid = {
            "n_estimators":
            trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth":
            trial.suggest_int("max_depth", 3, 12),
            "min_samples_split":
            trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":
            trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":
            trial.suggest_categorical("max_features",
                                      ['auto', 'sqrt', 'log2']),
            "bootstrap":
            trial.suggest_categorical("bootstrap", [True, False]),
            "criterion":
            trial.suggest_categorical("criterion", ['gini', 'entropy']),
        }
        return RandomForestClassifier(**param_grid)
