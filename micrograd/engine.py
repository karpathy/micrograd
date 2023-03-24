class Value:
    """stores a single scalar value and its gradient
    >>> x = Value(3.0)
    >>> y = x * x
    >>> y.backward()
    >>> x.grad
    6.0
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        """Add two values"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            """backpropagate the gradient to the inputs"""
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """multiply two values"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """raise a value to a power"""
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """relu activation function"""
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """backpropagate the gradient through the graph"""
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        """negate a value"""
        return self * -1

    def __radd__(self, other):  # other + self
        """add two values"""
        return self + other

    def __sub__(self, other):  # self - other
        """subtract two values"""
        return self + (-other)

    def __rsub__(self, other):  # other - self
        """subtract two values"""
        return other + (-self)

    def __rmul__(self, other):  # other * self
        """multiply two values"""
        return self * other

    def __truediv__(self, other):  # self / other
        """divide two values"""
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        """divide two values"""
        return other * self**-1

    def __repr__(self):
        """print a value"""
        return f"Value(data={self.data}, grad={self.grad})"
