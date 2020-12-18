
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', _backward=None):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._backward = _backward

    def __add__(self, other):
        if not isinstance(other, Value): 
            other = Value(other)
        return Value(self.data + other.data, (self, other), '+', lambda _: (1, 1))

    def __mul__(self, other):
        if not isinstance(other, Value): 
            other = Value(other)
        return Value(self.data * other.data, (self, other), '*', lambda n: (n._prev[1].data, n._prev[0].data))

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        return Value(self.data**other, (self,), f'**{other}', lambda n: (other * n._prev[0].data**(other-1),))

    def relu(self):
        return Value(max(0, self.data), (self,), 'ReLU', lambda n: (n.data > 0,))

    def _toposort(self):
        postorder = []
        visited = set()
        def dfs(node):
            if node._prev and node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                postorder.append(node)
        dfs(self)
        return reversed(postorder)

    def backward(self):
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in self._toposort():
            for child, local_grad in zip(v._prev, v._backward(v)):
                child.grad += local_grad * v.grad

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
