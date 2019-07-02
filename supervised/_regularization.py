from enum import Enum
import numpy as np


class REGULARIZE(Enum):
    L1 = 0
    L2 = 1


class Regularizer(object):
    """Imoplementation of regularization class."""

    def __init__(self, choice: REGULARIZE):
        self.__choice = choice
        self.__constant = 100.0  # Fixed shrinkage parameter

    def loss(self, w: np.ndarray) -> float:
        if self.__choice == REGULARIZE.L1:
            res = self.__constant * float(np.sum(np.abs(w)))
        elif self.__choice == REGULARIZE.L2:
            res = self.__constant * float(np.sum(w.dot(w)))
        else:
            raise NotImplementedError
        return res

    def grad(self, w: np.ndarray) -> np.ndarray:
        if self.__choice == REGULARIZE.L1:
            res = self.__constant * np.sign(w)
        elif self.__choice == REGULARIZE.L2:
            res = self.__constant * 2 * w
        else:
            raise NotImplementedError
        return res
