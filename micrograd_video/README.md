# Micrograd Lecture Companion Notebook

***Generated with GitHub Copilot***

This Jupyter notebook named micrograd_video_companion.ipynb is associated with a lecture given by Andrej Karpathy. The URL of this lecture is provided at the beginning of the notebook.

## Overview
This companion notebook aims to provide a step-by-step granular explanation and code implementation of backpropagation and neural network training. It employs Python and requires only a basic understanding of Python programming and some rudimentary recollections of calculus.

## Structure and Content
The notebook starts by importing the required Python libraries:

- random: provides functions to generate random numbers.
- numpy: allows working with arrays and provides mathematical functions.
- matplotlib: a library for creating static, animated and interactive plots in Python.

## Demonstration of Derivative Calculation:
The notebook demonstrates derivative estimation with code snippets. Initially, it opens with introducing a simple function f(x) = 3x^2 - 4x + 5. This function is defined using Python syntax def f(x): return 3*x**2 - 4*x+5. The value of the function is computed at x = 3.0.

The function is then plotted using the matplotlib library. This results in a visual representation of the function f.

The derivative of f at x = 2/3 is estimated using the limit definition of a derivative. This is done via the approximation f'(x) = (f(x+h)-f(x))/h where h is a small number (h = 0.0001 in this case).

## Evaluating Derivative of a Function with Multiple Inputs
The notebook advances into a function with multiple inputs, defining the function d1 = a*b + c, with a, b, and c as respective inputs.

The derivative calculation is done in a similar manner to above. The function's output is determined for slightly varied values of a, b, and c with the results subsequently plotted on a graph.

## More sections to describe the notebook based on the content that follows
TBD....

## How to Use This Notebook
Open the Jupyter notebook and follow along with the comments and markdown text to understand the purpose of each code cell. Run each cell to see the output and get a feel for how the mathematical concepts translate into code and visualizations. For a deeper understanding, experiment with the functions and derivative calculations by changing the values of x, a, b, and c.

Remember to have the necessary libraries installed and imported before running the notebook. If you don't have them installed, use pip install command on your terminal to install them first.

## Dependencies
- Python 3+
- numpy
- matplotlib
- torch
- random
