
# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :D). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are currently about 50 lines of code each.

**super lite**. The DAG only allows individual scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. In particular, the current library only supports scalars and three operations over them: (+,*,relu), but in fact these are enough to build up an entire deep neural net doing binary classification, as the demo notebook shows.

Potentially useful for educational purposes. See the notebook for demo.

