import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        other_grad = True
        
        if not isinstance(other, Value):
            other_grad = False
            other = Value(other)
        
        result = Value(self.data ** other.data , (self,other) , '**')
        
        def _backward():
            self.grad += (other.data) * ((self.data) ** (other.data - 1)) * (result.grad) 
            if other_grad : 
                other.grad += round((self.data ** other.data) * math.log(self.data),4)
            
        result._backward = _backward
        
        return result

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

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

    def __rpow__(self, other):
        other_grad = True
        if not isinstance(other , Value):
            other_grad = False
            other = Value(other)
            
        result = Value(other.data ** self.data , (self,other) , '**')
        
        def _backward():
            self.grad = round((other.data ** self.data) * math.log(other.data) , 4)
            if other_grad:
                other.grad += (other.data) * ((self.data) ** (other.data - 1)) * (result.grad)
            
        result._backward = _backward
        
        return result

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"