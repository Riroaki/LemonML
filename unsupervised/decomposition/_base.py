import numpy as np
from abc import ABC, abstractmethod


class Decomposition(ABC):
    """Abstract class for decomposition of data.
    Define basic operations of decomposition here:
        reduce.
    """

    @staticmethod
    @abstractmethod
    def extract_feature(x: np.ndarray, k: int = None) -> tuple:
        """Get reduced eigen vectors and eigen values.

        :param x: input data to reduce dimension, shape = (n, p)
        :param k: number of reduced dimensions, default means p
        :return: sorted eigen values, eigen vectors (large to small)
        shape of eigen vectors: (p, k)
        """

    def reduce(self, x: np.ndarray, k: int = None) -> np.ndarray:
        """Project x into lower dimensions using extracted features.

        :param x: input data to reduce dimension, shape = (n, p)
        :param k: number of reduced dimensions, default means p
        :return: projections of x in reduced dimensions
        shape of projection: (n, p)
        """
        _, feature = self.extract_feature(x, k)
        projection = np.matmul(x, feature)
        return projection
