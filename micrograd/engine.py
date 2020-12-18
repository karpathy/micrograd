
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = None
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(_):
            return (1, 1)
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(n):
            return (n._prev[1].data, n._prev[0].data)
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward(n):
            return (other * n._prev[0].data**(other-1),)
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward(n):
            return (n.data > 0,)
        out._backward = _backward

        return out

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
