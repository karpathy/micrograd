
# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :D). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are currently about 50 lines of code each.

The DAG only allows individual scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. In particular, the current library only supports scalars and three operations over them: (+,*,relu), but in fact these are enough to build up an entire deep neural net doing binary classification, as the demo notebook shows.

### Example usage

```python
from micrograd.engine import Value

x = Value(1.0)
z = 2 * x + 2 + x
q = z + z * x
h = z * z
y = z * z + q + q * x
print(y.data) # prints 45.0
y.backward()
print(x.grad) # prints 62.0 - i.e. the numerical value of dy / dx
```

Potentially useful for educational purposes. See the notebook for a full demo of training an MLP binary classifier.




