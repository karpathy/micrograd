
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', requires_grad=False, gradient_fn=None):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.requires_grad = requires_grad
        self.gradient_fn = None if not gradient_fn else lambda: gradient_fn(self)

    def __add__(self, other):
        if not isinstance(other, Value): 
            other = Value(other)
        requires_grad = self.requires_grad or other.requires_grad
        gradient_fn = None if not requires_grad else lambda _: (1, 1)
        return Value(self.data + other.data, (self, other), '+', requires_grad, gradient_fn)

    def __mul__(self, other):
        if not isinstance(other, Value): 
            other = Value(other)
        requires_grad = self.requires_grad or other.requires_grad
        gradient_fn = None if not requires_grad else lambda v: (v._prev[1].data, v._prev[0].data)
        return Value(self.data * other.data, (self, other), '*', requires_grad, gradient_fn)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        gradient_fn = None if not self.requires_grad else lambda v: (other * v._prev[0].data**(other-1),)
        return Value(self.data**other, (self,), f'**{other}', self.requires_grad, gradient_fn)

    def relu(self):
        gradient_fn = None if not self.requires_grad else lambda v: (v.data > 0,)
        return Value(max(0, self.data), (self,), 'ReLU', self.requires_grad, gradient_fn)

    def _toposort(self):
        postorder = []
        visited = set()
        def dfs(node):
            if node.requires_grad and node._prev and node not in visited:
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
            for child, local_grad in zip(v._prev, v.gradient_fn()):
                if child.requires_grad:
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
