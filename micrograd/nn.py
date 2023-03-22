import random

from micrograd.engine import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        """Sets the gradient of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """Returns a list of all parameters in the module."""
        return []


class Neuron(Module):
    """A single neuron with `nin` inputs."""

    def __init__(self, nin, nonlin=True):
        """Creates a neuron with `nin` inputs."""
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """Computes the output of the neuron for the given input."""
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """Returns a list of all parameters in the module."""
        return self.w + [self.b]

    def __repr__(self):
        """Returns a string representation of the neuron."""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """A layer of `nout` neurons with `nin` inputs."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Computes the output of the layer for the given input."""
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Returns a list of all parameters in the module."""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """Returns a string representation of the layer."""
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A multi-layer perceptron with `nin` inputs and `nouts` outputs."""

    def __init__(self, nin, nouts):
        """Creates an MLP with `nin` inputs and `nouts` outputs.

        Args:
            nin: number of inputs
            nouts: list of numbers of outputs for each layer
        """
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """Computes the output of the MLP for the given input.
        Args:
            x: input tensor
        Returns:
            output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Returns a list of all parameters in the module.
        Returns:
            list of parameters
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """Returns a string representation of the MLP.
        Returns:
            string representation
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
