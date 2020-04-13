
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data):
        self.data = data
        self.grad = 0
        self.backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)

        def backward():
            self.grad += out.grad
            other.grad += out.grad
            self.backward()
            other.backward()
        out.backward = backward

        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            self.backward()
            other.backward()
        out.backward = backward

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def relu(self):
        out = Value(0 if self.data < 0 else self.data)
        def backward():
            self.grad += (out.data > 0) * out.grad
            self.backward()
        out.backward = backward
        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
