import numpy as np
import pickle
from abc import abstractmethod, ABC


class Model(ABC):
    """Abstract class for models of both regression and classifying.
    Define basic operations of a model here:
        fit, predict, evaluate, and dump or load.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.float:
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
        :return: regression values
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


class LinearModel(Model, ABC):
    """Abstract class for linear models, adding variables and related methods
    ONLY for inner calls.
    """

    def __init__(self):
        """Define parameters used in linear models here."""
        self._w = None
        self._b = None
        self._optimum = None
        self._loss_tol = 1e-6
        self._batch_size = 64
        self._iter_bound = 2000
        # Important value
        self._learn_rate = 3e-3

    def _init_model(self, p: int):
        """Randomly initialize weights and bias,
        and params for best performance.
        """
        self._w = np.random.rand(p) * np.sqrt(1 / p)
        self._b = 0.
        self._optimum = {'loss': np.inf,
                         'w': None, 'b': None}

    def _update_model(self, loss: np.float):
        """Update model when current loss is smaller
        than that of optimal one."""
        if loss < self._optimum['loss']:
            self._optimum['loss'] = loss
            self._optimum['w'] = self._w
            self._optimum['b'] = self._b

    def _update_params(self, params: dict):
        """Update parameters using user's kwargs here."""
        for name, value in params.items():
            # WARN: this might be potentially risky
            inner_name = '_{}__{}'.format(type(self).__name__, name)
            assert inner_name in self.__dict__
            assert isinstance(value, type(self.__dict__[inner_name]))
            setattr(self, inner_name, value)

    @staticmethod
    @abstractmethod
    def _predict_value(x: np.ndarray, w: np.ndarray,
                       b: np.float) -> np.ndarray:
        """Calculate values of prediction using y = wx + b.
        Logistic regression uses sigmoid(wx + b).

        :param x: input data
        :return: prediction values
        """

    @staticmethod
    @abstractmethod
    def _predict_label(pred_val: np.ndarray) -> np.ndarray:
        """Calculate labels given predictions.
        Used in classification only.

        :param pred_val: values of prediction
        :return: predicted labels
        """

    @staticmethod
    @abstractmethod
    def _loss(pred_val: np.ndarray, true_val: np.ndarray) -> np.float:
        """Calculate loss value of predictions.

        :param pred_val: prediction values
        :param true_val: true values / labels
        :return: loss value
        """

    @abstractmethod
    def _grad(self, x: np.ndarray, pred_val: np.ndarray,
              true_val: np.ndarray) -> tuple:
        """Calculate the gradients for weights and bias.

        :param x: input x
        :param pred_val: prediction values
        :param true_val: true values
        :return: gradients for weights and bias of model
        """
