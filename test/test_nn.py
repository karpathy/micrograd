import unittest

from micrograd.engine import Value
from micrograd.nn import Neuron


class TestNeuron(unittest.TestCase):
    def test_adds_bias_before_activation(self):
        cases = [
            (False, 5.0, 3.0, 1.0),
            (False, 1.0, -1.0, 1.0),
            (True, 5.0, 3.0, 1.0),
            (True, 1.0, 0.0, 0.0),
        ]
        for nonlin, bias, expected, gradient in cases:
            with self.subTest(nonlin=nonlin, bias=bias):
                neuron = Neuron(2, nonlin=nonlin)
                neuron.w = [Value(2.0), Value(-3.0)]
                neuron.b = Value(bias)
                inputs = [Value(2.0), Value(2.0)]

                output = neuron(inputs)

                self.assertIsInstance(output, Value)
                self.assertEqual(output.data, expected)

                output.backward()

                self.assertEqual(neuron.b.grad, gradient)
                self.assertEqual(
                    [weight.grad for weight in neuron.w], [2 * gradient, 2 * gradient]
                )
                self.assertEqual(
                    [value.grad for value in inputs], [2 * gradient, -3 * gradient]
                )
