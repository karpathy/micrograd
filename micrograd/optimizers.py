from typing import List
from micrograd.engine import Value

class SGD:
    '''
    Stochastic Gradient Descent - initial simple version without momentum, etc.
    '''

    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params: List[Value]):
        for p in params:
            p.data -= self.lr * p.grad