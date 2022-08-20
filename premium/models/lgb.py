#!/usr/bin/env python
import abc

import codefast as cf
import joblib
import lightgbm as lgb
import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.model_selection import KFold

import premium as pm

class _Base:
    def __init__(self) -> None:
        pass
    
    def build_model(self, extra_parameters: dict = {}, enable_gpu=True):
        ''' Args: 
        enable_gpu:boolean, training with GPU or CPU.
        extra_parameters:dict, alter default parameters of model
        model_type:str, C for classifier, and R for regressor
        '''
        cf.info('Build lightgbm model with extra parameters', extra_parameters)
        self.default_params.update(extra_parameters)
        if enable_gpu:
            cf.info('Enable training with GPU')
            _extra = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
            self.default_params.update(_extra)
        else:
            cf.info('Enable training with CPU')
            self.default_params['device'] = 'cpu'



class LightgbmRegressor(_Base):
    def __init__(self, X, y, X_test):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.test_size = 0.2
        self.model_type = 'lightgbm_regressor'
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 20000,
            'random_state': 2021,
            'learning_rate': 0.001,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.6,
            'reg_alpha': 6.4,
            'reg_lambda': 1.8,
            'min_child_weight': 256,
            'min_child_samples': 20,
            'importance_type': 'gain',
            'device': 'gpu',
            'first_metric_only': True
        }

    def build_model(self, extra_parameters: dict = {}, enable_gpu=True):
        super(LightgbmRegressor, self).build_model(extra_parameters, enable_gpu)
        self.model = lgb.LGBMRegressor(**self.default_params)
        cf.info('LightGBM Regressor model successfully built.')
        return self.model

    def k_fold(self,
               n_splits: int = 10,
               random_state: int = 2021,
               metric=pm.libra.rmse,
               eval_metric: str = 'auc'):
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

        for fold, (trn_idx, val_idx) in enumerate(self.kf.split(self.X,
                                                                self.y)):
            cf.info(f'........... FOLD {fold} ............')
            X_train, y_train = self.X.iloc[trn_idx], self.y.iloc[trn_idx]
            X_valid, y_valid = self.X.iloc[val_idx], self.y.iloc[val_idx]

            _model_type = str(type(self.model))
            cf.info('modle type', _model_type)
            self.model.fit(X_train,
                           y_train,
                           eval_set=[(X_valid, y_valid)],
                           eval_metric=eval_metric,
                           early_stopping_rounds=200,
                           verbose=1000)
            oof[val_idx] = self.model.predict(X_valid)
            y_pred += self.model.predict(self.X_test)
            score = metric(y_valid, oof[val_idx])
            average_score += score / n_splits
            cf.info(f"fold {fold} - score: {score:.6f}")
        y_pred /= n_splits
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


class LightgbmClassifier:
    '''lightgbm Classifier
    '''
    def __init__(self, X, y, X_test):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.test_size = 0.2
        self.model_type = 'lightgbm_classifier'
        self.default_params = {
            'seed': 2021,
            'objective': 'binary',
            'num_leaves': 256,
            'subsample': 0.6,
            'subsample_freq': 1,
            'colsample_bytree': 0.4,
            'reg_alpha': 15.0,
            'reg_lambda': 1e-1,
            'min_child_weight': 256,
            'min_child_samples': 64,
            # 'importance_type': 'gain',
            'learning_rate': 0.002,
            # 'early_stopping_round': 700,
            'num_iterations': 50000,
            'device': 'gpu'
        }

    def build_model(self, extra_parameters: dict = {}, enable_gpu=True):
        _Base.build_model(self, extra_parameters, enable_gpu)
        self.model = lgb.LGBMClassifier(**self.default_params)
        cf.info('LightGBM Classifier model successfully built.')
        return self.model

    def k_fold(self,
               n_splits: int = 10,
               random_state: int = 2021,
               metric=pm.libra.rmse,
               eval_metric: str = 'auc',
               predict_proba: bool = False):
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
        # Make binary predition or probability prediction.
        func_predict = self.model.predict_proba if predict_proba else self.model.predict

        for fold, (trn_idx, val_idx) in enumerate(self.kf.split(self.X,
                                                                self.y)):
            cf.info(f'........... FOLD {fold} ............')
            X_train, y_train = self.X.iloc[trn_idx], self.y.iloc[trn_idx]
            X_valid, y_valid = self.X.iloc[val_idx], self.y.iloc[val_idx]

            _model_type = str(type(self.model))
            cf.info('modle type', _model_type)
            self.model.fit(X_train,
                           y_train,
                           eval_set=[(X_valid, y_valid)],
                           eval_metric=eval_metric,
                           early_stopping_rounds=700,
                           verbose=1000)
            oof[val_idx] = func_predict(X_valid)[:, -1]
            y_pred += func_predict(self.X_test)[:, -1]
            score = metric(y_valid, oof[val_idx])
            average_score += score / n_splits
            cf.info(f"fold {fold} - score: {score:.6f}")
        y_pred /= n_splits
        cf.info('Average score', average_score)

        return y_pred, average_score

    def fit(self, X, y):
        # X_train, y_train, X_valid, y_valid=
        cf.info('start training with model', self.model)
        self.model.fit(X,y)
        cf.info(f'Training completed')

        if self.save_model:
            model_name = cf.io.tmpfile(self.model_type, 'joblib')
            joblib.dump(self.model, model_name, compress=9)
            cf.info('model saved to {}'.format(model_name))

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
