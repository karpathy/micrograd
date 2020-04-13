
# MicroGrad

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :D). Implements backpropagation over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API.

The amusing part is that the DAG only allows individual scalar values, so e.g. we chop up every neuron into all of its individual tiny adds and multiplies. In particular, the current library only supports scalars and three operations: +,*,relu, but these are enough to build up an entire deep neural net doing binary classification. It's just a lot of nodes :)

Potentially useful for educational purposes. See the notebook for demo.

