
# MicroGrad

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :D). Implements backpropagation over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are currently about 50 lines of code each.

The amusing part is that the DAG only allows individual scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. In particular, the current library only supports scalars and three operations over them: (+,*,relu), but these are actually enough to build up an entire deep neural net doing binary classification as the demo notebook shows. It's just a lot of nodes :)

Potentially useful for educational purposes. See the notebook for demo.

