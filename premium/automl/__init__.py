import codefast as cf
from flaml import AutoML


def automl_fit_predict(train, labels, test, **kwargs):
    cf.info('start automl')
    automl = AutoML()
    settings = {
        'time_budget': 23690,
        'metric': 'accuracy',
        'task': 'classification',
        'log_file_name': '/tmp/automl.log',
    }
    settings.update(kwargs)
    automl.fit(train, labels, **settings)
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print('Best ap on validation data: {0:.4g}'.format(1 - automl.best_loss))
    print('Training duration of best run: {0:.4g} s'.format(
        automl.best_config_train_time))
    preds = automl.predict(test)
    return preds
