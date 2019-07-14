import numpy as np
import pickle
from abc import abstractmethod, ABC


class Layer(ABC):
    """Abstract class for models of layers.
    Define basic operations here:
        forward, backward, update, dump, load
    """

    @abstractmethod
    def forward(self, in_arr: np.ndarray) -> np.ndarray:
        """Forward calculation.

        :param in_arr: input array
        :return: output array
        """

    @abstractmethod
    def backward(self, sensitivity: np.ndarray) -> np.ndarray:
        """Backpropagation calculation.

        :param sensitivity: gradient passed from last layer
        :return: sensitivity of this layer
        """

    @abstractmethod
    def update(self, learn_rate: float) -> None:
        """Update weights and bias after calling backward function.

        :param learn_rate: learning rate of gradient descent
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


class Activation(ABC):
    """Abstract class for activation functions.
    Define basic operations here:
        forward, backward
    """

    @abstractmethod
    def forward(self, in_arr: np.ndarray) -> np.ndarray:
        """Forward calculation.

        :param in_arr: input array
        :return: output array
        """

    @abstractmethod
    def backward(self, out: np.ndarray) -> np.ndarray:
        """Backpropagation calculation.

        :param out: output of this layer
        :return: sensitivity of this layer
        """


class Criterion(ABC):
    """Abstract class for criterion of model output (Actually, loss function).
    Define basic operations here:
        forward, backward
    """

    @abstractmethod
    def forward(self, pred: np.ndarray, y: np.ndarray) -> float:
        """Forward calculation.

        :param pred: prediction values
        :param y: true values, regression / classification value
        :return: loss value
        """

    @abstractmethod
    def backward(self, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Backpropagation calculation.

        :param pred: prediction values
        :param y: true values, regression / classification value
        :return: derivative of loss function
        """


class Module(ABC):
    """Abstract class for neural network models.
    Define basic operations here:
        train, predict, dump, load
    """

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Train model and update model using gradient descent.
        Should preserve optimum model after loss is calculated.

        :param x: input data, shape = (n, dim)
        :param y: true values / labels of regression results, shape = (n,)
        :return: total training loss
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Calculate predicting values of given input.

        :param x: input x
        :return: predicting values
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
