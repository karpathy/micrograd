import torch
from torch import nn
import sys
import os
from typing import List
import random

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value
from micrograd.nn import Linear

def convert_micrograd_values_to_torch_values(micrograd_values: List[Value]) -> torch.Tensor:
    values = [v.data for v in micrograd_values]
    return torch.tensor(values, dtype=torch.float64)

def convert_torch_values_to_micrograd_values(torch_values: torch.Tensor) -> List[Value]:
    values = torch_values.tolist()
    return [Value(v) for v in values]

def convert_linear_micrograd_weights_to_torch_tensor(micrograd_weights: Linear) -> torch.Tensor:
    # get shape of weight matrix
    shape = (len(micrograd_weights), len(micrograd_weights[0]))
    weights = []

    # convert micrograd weights to torch weights
    for neuron_weights in micrograd_weights:
        assert len(neuron_weights) == shape[1]
        weights.append(convert_micrograd_values_to_torch_values(neuron_weights))

    weights = torch.stack(weights)
    return weights

def transfer_weights_from_micrograd_to_torch(micrograd_linear: Linear, torch_linear: nn.Linear, use_bias: bool) -> None:
    # convert micrograd weights to torch weights
    weights = convert_linear_micrograd_weights_to_torch_tensor(micrograd_linear.get_weights())
    weights = nn.Parameter(weights)
    torch_linear.weight = weights
    # convert micrograd biases to torch biases
    if use_bias:
        biases = convert_micrograd_values_to_torch_values(micrograd_linear.get_biases())
        biases = nn.Parameter(biases)
        torch_linear.bias = biases

def compare_micrograd_and_torch_value(micrograd_value: Value, torch_value: torch.Tensor, precision=1e-3) -> bool:
    return abs(micrograd_value.data - torch_value.item()) < precision

def compare_micrograd_and_torch_gradient(micrograd_value: Value, torch_value: torch.Tensor, precision=1e-3) -> bool:
    return abs(micrograd_value.grad - torch_value) < precision

def compare_micrograd_and_torch_values(micrograd_values: List[Value], torch_values: torch.Tensor, precision=1e-3) -> bool:
    assert len(micrograd_values) == len(torch_values)
    for micrograd_value, torch_value in zip(micrograd_values, torch_values):
        if not compare_micrograd_and_torch_value(micrograd_value, torch_value, precision=precision):
            return False
    return True

def compare_micrograd_and_torch_gradients(micrograd_values: List[Value], torch_values: torch.Tensor, precision=1e-3) -> bool:
    assert len(micrograd_values) == len(torch_values)
    for micrograd_value, torch_value in zip(micrograd_values, torch_values):
        if not compare_micrograd_and_torch_gradient(micrograd_value, torch_value, precision=precision):
            return False
    return True

def compare_linear_micrograd_and_torch_gradients(micrograd_weights: List[List[Value]], torch_weights: torch.Tensor, 
                                                 micrograd_biases: List[Value]=None, torch_biases: torch.Tensor=None, precision=1e-3
                                                 ) -> bool:
    # get shape of weight matrix
    micrograd_shape = (len(micrograd_weights), len(micrograd_weights[0]))
    torch_shape = torch_weights.shape
    assert micrograd_shape == torch_shape

    # compare micrograd weight gradients to torch weight gradinets
    for neuron_micrograd_weights, neuron_tensor_grads in zip(micrograd_weights, torch_weights.grad):
        assert len(neuron_micrograd_weights) == micrograd_shape[1]
        assert len(neuron_tensor_grads) == torch_shape[1]
        assert len(neuron_micrograd_weights) == len(neuron_tensor_grads)
        if not compare_micrograd_and_torch_gradients(neuron_micrograd_weights, neuron_tensor_grads, precision=precision):
            return False

    # compare micrograd bias gradients to torch bias gradients
    if micrograd_biases is not None and torch_biases is not None:
        if not compare_micrograd_and_torch_gradients(micrograd_biases, torch_biases.grad, precision=precision):
            return False

    return True


def test_linear_sample(in_neurons: int, out_neurons: int, nonlin=False, use_bias=True, precision=1e-3, verbose=False) -> bool:
    # generate random input data array
    x_torch = torch.randn(in_neurons, dtype=torch.float64)
    # convert x to micrograd Value
    x_micrograd = convert_micrograd_values_to_torch_values(x_torch)

    # micrograd code
    micrograd_linear = Linear(in_neurons, out_neurons, nonlin=nonlin ,use_bias=use_bias)

    if verbose:
        print("Micrograd weights")
        print(micrograd_linear.get_weights())
        if use_bias:
            print(micrograd_linear.get_biases())
        print("Micrograd output")
    micrograd_out = micrograd_linear(x_micrograd)
    if verbose:
        print(micrograd_out)

    # backward pass
    for m_out in micrograd_out:
        m_out.backward()

    if verbose:
        print("Micrograd weights after backward pass")
        print(micrograd_linear.get_weights())
        if use_bias:
            print(micrograd_linear.get_biases())

    # torch code
    if verbose:
        print("Input")
        print(x_torch.tolist())
    torch_linear = torch.nn.Linear(in_neurons, out_neurons, bias=use_bias, dtype=torch.float64)

    if verbose:
        print("Pytorch weights before transfer")
        print(torch_linear.parameters())
        print(torch_linear.weight)
        if use_bias:
            print(torch_linear.bias)


    transfer_weights_from_micrograd_to_torch(micrograd_linear, torch_linear, use_bias=use_bias)
    if verbose:
        print("Pytorch weights")
        print(torch_linear.parameters())
        print(torch_linear.weight)
        if use_bias:
            print(torch_linear.bias)
        print("Pytorch output")

    torch_out = torch_linear(x_torch)
    if nonlin:
       torch_out = nn.ReLU()(torch_out)
    if verbose:
        print(torch_out)

    # backward pass
    for t_out in torch_out:
        t_out.backward(retain_graph=True)

    if verbose:
        print("Pytorch gradients after backward pass")
        print(torch_linear.parameters())
        print(torch_linear.weight.grad)
        if use_bias:
            print(torch_linear.bias.grad)

    if compare_micrograd_and_torch_values(micrograd_out, torch_out, precision):
        print("Forward test passed!")
    else: 
        print("Forward test failed!")
        return False
    
    if compare_linear_micrograd_and_torch_gradients(micrograd_linear.get_weights(), torch_linear.weight, 
                                                    micrograd_linear.get_biases() if use_bias else None,
                                                    torch_linear.bias if use_bias else None,
                                                    precision):
        print("Backward test passed!")
    else:
        print("Backward test failed!")
        return False

    return True

def linear_unit_test(num_tests: int = 100):
    correct_count = 0
    for i in range(num_tests):
        in_neurons = random.randint(1, 100)
        out_neurons = random.randint(1, 100)
        nonlin = random.choice([True, False])
        use_bias = random.choice([True, False])
        precision = 1e-3
        if test_linear_sample(in_neurons=in_neurons, 
                              out_neurons=out_neurons, 
                              nonlin=nonlin, 
                              use_bias=use_bias, 
                              precision=precision, 
                              verbose=False):
            correct_count += 1
    print(f"Passed {correct_count}/{num_tests} tests")

if __name__ == '__main__':
    linear_unit_test()  

