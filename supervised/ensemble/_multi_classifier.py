from enum import Enum
import numpy as np
from .._base import SupervisedModel
from .._logistic_regression import LogisticRegression
from .._support_vector_machine import SVM
from .._perceptron import Perceptron


class MULTICLS(Enum):
    ONE_VERSUS_ONE = 0
    ONE_VERSUS_REST = 1


class MultiClassifier(SupervisedModel):
    """Additional class for multi-classification support."""

    __binary_classifiers = {LogisticRegression, SVM, Perceptron}

    def __init__(self, cls: type, option: MULTICLS):
        # Only for binary classifiers
        assert cls in self.__binary_classifiers
        self.__option = option
        self.__cls = cls
        self.__categories = None
        self.__models = {}

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        # Get catogorical variables
        categories = np.unique(label)
        k = len(categories)
        # Reinitialize models if categories are not all recorded
        if not np.isin(False, np.isin(categories, self.__categories)):
            self.__models = {}
            self.__categories = categories
        total_loss = 0.
        if self.__option == MULTICLS.ONE_VERSUS_ONE:
            # One-vs-One
            # Choose pairs and form k(k - 1) / 2 classifiers.
            for i in range(k):
                for j in range(i + 1, k):
                    ci, cj = categories[i], categories[j]
                    model: SupervisedModel = self.__cls()
                    # Select pair of two class
                    x_pair = x[label == ci or label == cj].copy()
                    label_pair = label[label == ci or label == cj].copy()
                    # Map label pair: (ci, cj) -> (1, -1)
                    label_pair[label_pair == ci] = 1
                    label_pair[label_pair == cj] = -1
                    # Feed model with data and save
                    total_loss += model.fit(x_pair, label_pair)
                    self.__models[ci][cj] = model
        elif self.__option == MULTICLS.ONE_VERSUS_REST:
            # One-vs-Rest
            # Choose one and mask all as rest
            # However, this is not implemented yet
            # because confidence is not implemented in base classifiers
            raise NotImplementedError()
        else:
            # No such operation
            raise NotImplementedError()
        return total_loss

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if self.__option == MULTICLS.ONE_VERSUS_ONE:
            # One-vs-One: vote
            k = len(self.__categories)
            vote_bins = np.zeros(x.shape[0], k)
            for i in range(k):
                for j in range(i + 1, k):
                    ci, cj = self.__categories[i], self.__categories[j]
                    model: SupervisedModel = self.__models[ci][cj]
                    y_pred = model.predict(x)
                    # Recover from masked value
                    vote_i = y_pred[y_pred == 1]
                    vote_j = -y_pred[y_pred == -1]
                    # Vote!
                    vote_bins[:, i] += vote_i
                    vote_bins[:, j] += vote_j
            # Predictions depends on max votes for each sample
            label_pred = self.__categories[np.argmax(vote_bins, axis=1)]
        elif self.__option == MULTICLS.ONE_VERSUS_REST:
            # One-vs-Rest:
            # Needs to implement confidence in base classifiers...
            raise NotImplementedError()
        else:
            # No such operation
            raise NotImplementedError()
        return label_pred

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        assert x.shape[0] == label.shape[0]
        pred_labels = self.predict(x)
        # Use 0-1 loss
        loss = np.count_nonzero(pred_labels != label)
        precision = 1 - loss / label.shape[0]
        return precision, loss
