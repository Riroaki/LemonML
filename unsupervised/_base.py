import numpy as np
from abc import abstractmethod, ABC


class UnsupervisedModel(ABC):
    """Abstract class for unsupervised models, majorly for clustering.
    Define basic operations of unsupervised models here:
        clustering.
    """

    @abstractmethod
    def clustering(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Clustering data into some groups.

        :param x: input data to group into clusters, shape = (n, p)
        :param kwargs: parameters like k for k-means and spectral,
        and training parameters like iter_bound, max_iter, etc.
        :return: assigned index for input data, shape = (n,)
        """
