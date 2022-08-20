from typing import Dict, List, Tuple

import codefast as cf
from rich.console import Console
from rich.table import Table

from premium.data.preprocess import TrainData
from premium.demo import ClassifierDemos
from premium.models.clf import Classifier
from codefast.utils import timethis
import numpy as np


class Benchmark(object):
    def __init__(self,
                 td: TrainData,
                 models: List,
                 name: str = 'benchmark') -> None:
        """
        :param models: List of models to benchmark
        """
        self.td = td
        self.models = models
        self.name = name

    def evaluate_model(self, model, cv: int = 5) -> Dict:
        """
        :param model: Model to benchmark
        """
        cf.info('Evaluating model {}'.format(model.__class__.__name__))
        clf = Classifier(model)
        scores, _ = clf.cv2(self.td.x, self.td.y, cv=cv)
        return {
            'name': model.__class__.__name__,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'cv_scores': scores,
        }

    @timethis
    def run(self, cv: int = 5, display_only: bool = False) -> 'Benchmark':
        """ cross validate all models over train data, 
        and present the results in a table.
        Args:
            cv: number of cross validation rounds
            display_only: if True, only display the results previously calculated
        """
        if not display_only:
            scores = {}  # model to score
            for model in self.models:
                model_name = model.__class__.__name__
                result = self.evaluate_model(model, cv=cv)
                scores[model_name] = result
                cf.js.write(result, f"/tmp/{self.name}_{model_name}.json")
        self.display()
        return self

    def display(self):
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", justify="left")
        table.add_column("Mean", justify="middle")
        table.add_column("Std", justify="middle")
        table.add_column("cv scores", justify="middle")
        cc: List[Tuple[str, Dict]] = []
        for model in self.models:
            model_name = model.__class__.__name__
            op = f"/tmp/{self.name}_{model_name}.json"
            cf.info('loading result from {}'.format(op))
            result = cf.js.read(op)
            cc.append((model_name, result))
        cc.sort(key=lambda x: x[1]['mean'], reverse=True)
        for model_name, result in cc:
            table.add_row(model_name, str(round(result['mean'], 4)),
                          str(round(result['std'], 4)),
                          str([round(e, 6) for e in result['cv_scores']]))
        console.print(table)


def get_classifiers(use_gpu: bool = False):
    demo = ClassifierDemos()
    return [
        demo.lightgbm_classifier(use_gpu=use_gpu),
        demo.xgboost_classifier(use_gpu=use_gpu),
        demo.catboost_classifier(use_gpu=use_gpu),
        demo.ada_boost_classifier(),
        demo.svm_classifier(),
        demo.random_forest_classifier(),
        demo.extra_tree_classifier(),
    ]
