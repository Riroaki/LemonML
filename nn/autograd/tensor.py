import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod

# Adapted from https://github.com/tiandiweizun/autodiff

# Types of data
data_type = Union[int, float, np.ndarray]


def _type_exception(data):
    return TypeError(
        'Unexpected data type: {}, data: {}'.format(type(data), data))


class Tensor(object):
    """Gradient container."""

    def __init__(self, data: data_type, parents: list = None, op: 'Op' = None,
                 grad=None):
        self.data = data
        self.parents = parents
        self.op = op
        self.grad = grad

    def __add__(self, other):
        if isinstance(other, Tensor):
            return _add.forward([self, other])
        if isinstance(other, data_type.__args__):
            return _add_const.forward([self, other])
        raise _type_exception(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return _sub.forward([self, other])
        if isinstance(other, data_type.__args__):
            return _add_const.forward([self, -other])
        raise _type_exception(other)

    def __rsub__(self, other):
        # Only when other is not Tensor type, this method will be called.
        if isinstance(other, data_type.__args__):
            return _rsub_const.forward([self, other])
        raise _type_exception(other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return _mul.forward([self, other])
        if isinstance(other, data_type.__args__):
            return _mul_const.forward([self, other])
        raise _type_exception(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return _div.forward([self, other])
        if isinstance(other, data_type.__args__):
            return _mul_const.forward([self, 1 / other])
        raise _type_exception(other)

    def __rtruediv__(self, other):
        # Only when other is not Tensor type, this method will be called.
        if isinstance(other, data_type.__args__):
            return _rdiv_const.forward([self, other])
        raise _type_exception(other)

    def __neg__(self):
        return self.__rsub__(0)

    def matmul(self, other):
        if isinstance(other, Tensor):
            return _mul_mat.forward([self, other])
        raise _type_exception(other)

    def sum(self):
        return _sum.forward([self])

    def mean(self):
        return _sum.forward([self]) / self.data.size

    def exp(self):
        return _exp.forward([self])

    def log(self):
        return _log.forward([self])

    def backward(self, grad: List[np.ndarray] = None):
        if grad is None:
            if isinstance(self.data, np.ndarray):
                grad = np.ones_like(self.data)
            else:
                grad = 1
            self.grad = grad
        if self.op:
            grad_list = self.op.backward(self.parents, grad)
            for idx, grad in enumerate(grad_list):
                tensor = self.parents[idx]
                tensor.grad += grad

    def zero_grad(self):
        if isinstance(self.data, np.ndarray):
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = 0.

    __radd__ = __add__
    __rmul__ = __mul__


class Op(ABC):
    """Operation between tensors."""

    @abstractmethod
    def forward(self, operands: list) -> Tensor:
        """Forward calculation."""

    @abstractmethod
    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        """Backward calculation."""


class Add(Op):
    """Tensor + tensor."""

    def forward(self, operands: list) -> Tensor:
        tensor1, tensor2 = operands
        return Tensor(tensor1.data + tensor2.data, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        return [grad, grad]


class AddWithConst(Op):
    """Tensor + constant."""

    def forward(self, operands: list) -> Tensor:
        tensor1, const = operands
        return Tensor(tensor1.data + const, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        return [grad]


class Sub(Op):
    """Tensor - tensor."""

    def forward(self, operands: list) -> Tensor:
        tensor1, tensor2 = operands
        return Tensor(tensor1.data - tensor2.data, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        return [grad, -grad]


class RSubWithConst(Op):
    """Const - tensor."""

    def forward(self, operands: list) -> Tensor:
        tensor1, const = operands
        return Tensor(const - tensor1.data, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        return [-grad]


class Multiply(Op):
    """Tensor * tensor."""

    def forward(self, operands: list) -> Tensor:
        tensor1, tensor2 = operands
        return Tensor(tensor1.data * tensor2.data, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        tensor1, tensor2 = operands
        return [tensor2.data * grad, tensor1.data * grad]


class MultiplyWithConst(Op):
    """Tensor * const."""

    def forward(self, operands: list) -> Tensor:
        tensor1, const = operands
        return Tensor(tensor1.data * const, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        tensor1, const = operands
        return [const * grad]


class MultiplyWithMatrix(Op):
    """Tensor.matmul(Tensor)."""

    def forward(self, operands: list) -> Tensor:
        tensor1, tensor2 = operands
        return Tensor(np.matmul(tensor1.data, tensor2.data), parents=operands,
                      op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        # AB = Y -> dA = dYB^T, dB = A^TdY
        tensor1, tensor2 = operands
        return [np.matmul(grad, tensor2.data.T),
                np.matmul(tensor1.data.T, grad)]


class Divide(Op):
    """Tensor / tensor."""

    def forward(self, operands: list) -> Tensor:
        tensor1, tensor2 = operands
        return Tensor(tensor1.data / tensor2.data, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        tensor1, tensor2 = operands
        return [grad / tensor2.data,
                -grad * tensor1.data / tensor2.data ** 2]


class RDivideWithConst(Op):
    """Const / tensor."""

    def forward(self, operands: list) -> Tensor:
        tensor, const = operands
        return Tensor(const / tensor, parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        tensor, const = operands
        return [-grad * const / tensor.data ** 2]


class Sum(Op):
    """Sum(tensor)."""

    def forward(self, operands: list) -> Tensor:
        tensor = operands[0]
        return Tensor(np.sum(tensor.data), parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        tensor = operands[0]
        return [grad * np.ones_like(tensor.data)]


class Exp(Op):
    """Exp(tensor)."""

    def forward(self, operands: list) -> Tensor:
        tensor = operands[0]
        return Tensor(np.exp(tensor.data), parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        tensor = operands[0]
        return [grad * np.exp(tensor.data)]


class Log(Op):
    """Log(tensor)."""

    def forward(self, operands: list) -> Tensor:
        tensor = operands[0]
        return Tensor(np.log(tensor.data), parents=operands, op=self)

    def backward(self, operands: list, grad: np.ndarray) -> List[np.ndarray]:
        tensor = operands[0]
        return [grad / tensor.data]


# Operator instances
_add = Add()
_add_const = AddWithConst()
_sub = Sub()
_rsub_const = RSubWithConst()
_mul = Multiply()
_mul_const = MultiplyWithConst()
_mul_mat = MultiplyWithMatrix()
_div = Divide()
_rdiv_const = RDivideWithConst()
_sum = Sum()
_exp = Exp()
_log = Log()
