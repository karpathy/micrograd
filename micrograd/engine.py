from typing import Iterable, Callable, Optional, Union

class Value:
    """ stores a single scalar value and its gradient """

    __slots__ = 'data', 'grad', '_backward', '_op', '_lhs', '_rhs', '_k', 'refresh'

    data: float
    grad: float
    _backward: Callable[[], None]
    _op: str
    _k: int
    refresh: Callable[[int], float]
    _lhs: Optional['Value']
    _rhs: Union['Value', float, int, None]

    def __init__(self, data: float, lhs=None, rhs=None, op: str=''):
        self.data = data
        self.grad = 0.0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._lhs = lhs
        self._rhs = rhs
        self._op = op # the op that produced this node, for graphviz / debugging / etc
        self._k = 0
        self.refresh = self.__refresh_value

    def assign(self, value: float) -> None:
        self.data = value
        self.grad = 0.0
        self._backward = lambda: None
        self._lhs = None
        self._rhs = None
        self._op = ''
        self._k = 0
        self.refresh = self.__refresh_value

    def __refresh_value(self, k: int) -> float:
        if self._k < k:
            self._k = k
            self.grad = 0.0
        return self.data

    def __refresh_add(self, k: int) -> float:
        if self._k >= k:
            return self.data
        lhs = self._lhs.refresh(k)
        rhs = self._rhs.refresh(k)
        self.data = value = lhs + rhs
        self._k = k
        self.grad = 0.0
        return value

    def __refresh_mul(self, k: int) -> float:
        if self._k >= k:
            return self.data
        lhs = self._lhs.refresh(k)
        rhs = self._rhs.refresh(k)
        self.data = value = lhs * rhs
        self._k = k
        self.grad = 0.0
        return value

    def __refresh_pow(self, k: int) -> float:
        if self._k >= k:
            return self.data
        lhs = self._lhs.refresh(k)
        rhs: float = self._rhs
        self.data = value = lhs ** rhs
        self._k = k
        self.grad = 0.0
        return value

    def __refresh_relu(self, k: int) -> float:
        if self._k >= k:
            return self.data
        lhs = self._lhs.refresh(k)
        self.data = value = 0 if lhs < 0 else lhs
        self._k = k
        self.grad = 0.0
        return value
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, self, other, '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        out.refresh = out.__refresh_add

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, self, other, '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        out.refresh = out.__refresh_mul

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, self, other, f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        out.refresh = out.__refresh_pow

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, self, None, 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        out.refresh = out.__refresh_relu

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if isinstance(v._rhs, Value):
                    build_topo(v._rhs)
                if isinstance(v._lhs, Value):
                    build_topo(v._lhs)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
