from enum import Enum
import numpy as np
from supervised._base import SupervisedModel
from supervised._logistic import LogisticRegression
from supervised._svm import SVM
from supervised._perceptron import Perceptron


class OPTION(Enum):
    ONE_VERSUS_ONE = 0
    ONE_VERSUS_REST = 1


class MultiClass(SupervisedModel):
    """Additional class for multi-classification support."""

    __binary_classifiers = {LogisticRegression, SVM, Perceptron}

    def __init__(self, cls: type, option: OPTION):
        # Only for binary classifiers
        assert cls in self.__binary_classifiers
        self.__option = option
        self.__cls = cls
        self.__models = {}

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.float:
        pass

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        pass
