import math
import random
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
from micrograd.jit import jit
# helps investigate segmentation faults
import faulthandler
faulthandler.enable()


def test_value():
    a = Value(4.0)
    b = Value(2.0)
    c = a + b  # 6.
    d = a + c  # 10.
    jd = jit(d)
    assert math.isclose(d.data, jd(), abs_tol=1e-04)


def test_neuron():
    n = Neuron(nin=1, nonlin=False)
    n.w = [2.]
    jn = jit(n)
    args = [10.0]
    assert math.isclose(n(args).data, jn(args), abs_tol=1e-04)


def test_layer():
    random.seed(10)
    l = Layer(nin=2, nout=1)
    jl = jit(l)
    args = [-30., -20.]
    assert math.isclose(l(args).data, jl(args), abs_tol=1e-04)


def test_mlp():
    random.seed(10)
    nn = MLP(nin=2, nouts=[1])
    jnn = jit(nn)
    args = [-30., -20.]
    assert math.isclose(nn(args).data, jnn(args), abs_tol=1e-04)


def test_mlp_complex():
    random.seed(10)
    nn = MLP(nin=2, nouts=[2, 1])
    jnn = jit(nn)
    args = [-30., -20.]
    assert math.isclose(nn(args).data, jnn(args), abs_tol=1e-04)
