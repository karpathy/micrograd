

def grad(f):
    """Gradient of a scalar function"""
    def _grad(*args, **kwargs):
        f(*args, **kwargs).backward() # forward and backward passes
        return [x.grad for x in args] # extract partial derivatives
    _grad.__name__ = f"grad({f.__name__})"
    return _grad


def jacrev(f):
    """Jacobian of a vector-valued function"""
    def _jac(*args, **kwargs):
        y = f(*args, **kwargs) # forward pass
        J = []
        for yi in y:
            for x in args:
                x.grad = 0.0
            yi.backward()
            J.append([x.grad for x in args])
        return J
    _jac.__name__ = f"grad({f.__name__})"
    return _jac


def vjp(f, x, v):
    """Vector-Jacobian-transpose product for vector-valued function"""
    y = f(*x)
    adj_vec = []
    for yi in y:
        for xi in x:  # need to figure out how to "reset" better
            xi.grad = 0.0
        yi.backward()
        adj_vec.append(sum([vi*xi.grad for (vi, xi) in zip(v, x)]))
    return adj_vec