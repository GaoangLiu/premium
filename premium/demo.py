import codefast as cf
import tensorflow as tf
import random


class ClassifierDemos(object):
    """ Some machine learning classifier demos for references.
    """

    def tf_classifier(self,
                      vocab_size: int = 10000,
                      embedding_dim=64,
                      max_length=1000):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=max_length),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                #   tf.keras.metrics.Precision(name='precision'),
                #   tf.keras.metrics.Recall(name='recall')
            ])
        return model

    def catboost_classifier(self,
                            loss_function: str = 'Logloss',
                            use_gpu: bool = False) -> 'CatBoostClassifier':
        import catboost as cb
        return cb.CatBoostClassifier(iterations=1000,
                                     learning_rate=0.1,
                                     eval_metric='Accuracy',
                                     loss_function=loss_function,
                                     random_seed=random.randint(0, 0x3f3f3f3f),
                                     verbose=True,
                                     early_stopping_rounds=50,
                                     task_type="GPU" if use_gpu else "CPU",
                                     metric_period=50,
                                     devices='0:1')

    def lightgbm_classifier(
            self,
            objective: str = 'binary',
            metric: str = 'binary_logloss',
            use_gpu: bool = False) -> 'LightGBMClassifier':
        import lightgbm
        return lightgbm.LGBMClassifier(
            boosting_type='gbdt',
            learning_rate=0.1,
            n_estimators=300,
            objective=objective,  # 'binary', 'multiclass'
            metric=metric,  # 'binary_logloss', 'multi_logloss'
            random_state=42,
            verbose=1,
            n_jobs=-1,
            device='gpu' if use_gpu else 'cpu')

    def xgboost_classifier(self,
                           objective: str = 'binary:logistic',
                           use_gpu: bool = False) -> 'XGBClassifier':
        import xgboost
        return xgboost.XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            # objective='multi:softmax', # 'binary:logistic', 'multi:softmax'
            objective=objective,
            random_state=42,
            verbose=True,
            tree_method='gpu_hist' if use_gpu else 'auto')

    def extra_tree_classifier(self) -> 'ExtraTreeClassifier':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=1500,
                                    max_depth=None,
                                    min_samples_split=2,
                                    n_jobs=-1,
                                    verbose=1,
                                    random_state=42)

    def random_forest_classifier(self) -> 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500,
                                      max_depth=None,
                                      min_samples_split=2,
                                      random_state=42)

    def logistic_regression(self) -> 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0,
                                  random_state=42,
                                  solver='liblinear',
                                  multi_class='ovr')

    def ridge_regression(self) -> 'Ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=42, solver='auto')

    def svm_classifier(self) -> 'SVC':
        from sklearn.svm import SVC
        return SVC(C=1.0, kernel='rbf', random_state=42)

    def decision_tree_classifier(self) -> 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=None,
                                      min_samples_split=2,
                                      random_state=42)

    def gradient_boosting_classifier(self) -> 'GradientBoostingClassifier':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=500,
                                          learning_rate=0.1,
                                          max_depth=None,
                                          random_state=42)

    def ada_boost_classifier(self) -> 'AdaBoostClassifier':
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(n_estimators=500,
                                  learning_rate=0.1,
                                  random_state=42)

    def voting_classifier(self) -> 'VotingClassifier':
        from sklearn.ensemble import VotingClassifier
        return VotingClassifier(
            estimators=[
                ('catboost', self.catboost_classifier()),
                ('lightgbm', self.lightgbm_classifier()),
                ('xgboost', self.xgboost_classifier()),
                ('adaboost', self.ada_boost_classifier()),
                # ('extra_tree', self.extra_tree_classifier()),
                # ('random_forest', self.random_forest_classifier()),
                # ('logistic_regression', self.logistic_regression()),
                ('ridge_regression', self.ridge_regression()),
                # ('svm', self.svm_classifier()),
                # ('gradient_boosting', self.gradient_boosting_classifier())
            ],
            voting='soft',
            verbose=True)
        # , weights=[1, 1, 1, 1, 1, 1, 1, 1, 1])


classifiers = ClassifierDemos()


class Demo(object):
    def __init__(self) -> None:
        self.classifiers = ClassifierDemos()


demo_object = Demo()
