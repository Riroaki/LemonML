import numpy as np
import pickle
from abc import abstractmethod, ABC


class SupervisedModel(ABC):
    """Abstract class for models of supervised models:
        regression and classifying.
    Define basic operations of supervised model here:
        fit, predict, evaluate, and dump or load.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Train model and update model using gradient descent.
        Should preserve optimum model after loss is calculated.

        :param x: input data, shape = (n, dim)
        :param y: true values / labels of regression results, shape = (n,)
        :param kwargs: parameters including learning rate,
        tolerance of gradients' sum, iteration bound, etc.
        :return: loss of predictions
        """

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate regression / classification values of given input.

        :param x: input x
        :param kwargs: parameters for prediction, like k for knn
        :return: prediction values
        """

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        """Evaluate model by calculating precision and loss of predictions of
        given x and true values.

        NOTE: For regression, return (None, loss);
        For classifiers, return (precision, loss).

        :param x: input x
        :param y: true values
        :param kwargs: parameters for evaluation, like k for knn
        :return: precision and loss of predictions
        """

    def dump(self, dump_file: str):
        """Dump model's parameters to a file."""
        with open(dump_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, dump_file: str):
        """Load model's parameters from a file."""
        with open(dump_file, 'rb') as f:
            params = pickle.load(f)
        self.__dict__.update(params)
