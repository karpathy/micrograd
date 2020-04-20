import torch
from micrograd.engine import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_higher_order():
    x = Value(3)
    y = x**3
    # This might seem redundant, but better we do this to ensure gradient is
    # 0 before backward pass.
    x.grad = 0
    y.backward()
    dy = x.grad
    assert dy.data == (x.data**2) * 3
    x.grad = 0
    dy.backward()
    d2y = x.grad
    assert d2y.data == x.data*6
    x.grad = 0
    d2y.backward()
    d3y = x.grad
    assert d3y.data == 6
    x.grad = 0
    d3y.backward()
    d4y = x.grad
    assert d4y.data == 0

def test_higher_order():
    x = Value(3)
    y = Value(2)
    f = 2*x*x*y+y*x
    x.grad = y.grad = 0
    f.backward()
    dfx = x.grad
    dfy = y.grad
    assert dfx.data == 4*x.data*y.data + y.data
    assert dfy.data == 2*x.data*x.data + x.data
    x.grad = y.grad = 0
    dfx.backward()
    dfxx = x.grad
    dfxy = y.grad
    x.grad = y.grad = 0
    dfy.backward()
    dfyx = x.grad
    dfyy = y.grad
    print(dfxx,dfyx,dfyy,dfxy)
    assert dfyx.data == dfxy.data == (4*x.data + 1)
    assert dfyy == 0
    assert dfxx.data == 4*y.data
