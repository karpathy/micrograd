from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
import mlir.dialects.arith as arith
import mlir.dialects.math as math
import mlir.dialects.func as func
from mlir.ir import Context, Location, InsertionPoint, Module
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir import ir
import sys
from typing import Union
import math
import ctypes
import random


class Compiler:
    def __init__(self, compiled_values={}):
        self.compiled_values = compiled_values

    def walk(self, value: Value):
        if value in self.compiled_values:
            return self.compiled_values[value]
        match value._op:
            case '':
                return arith.constant(
                    value=float(
                        value.data),
                    result=ir.F32Type.get())
            case '*':
                lhs, rhs = value._prev
                return arith.mulf(self.walk(lhs), self.walk(rhs))
            case '+':
                lhs, rhs = value._prev
                return arith.addf(self.walk(lhs), self.walk(rhs))
            case 'ReLU':
                (item,) = value._prev
                return arith.maximumf(self.walk(Value(0.0)), self.walk(item))
        if "**" in value._op:
            base, exp = value._prev
            return math.powf(self.walk(base), self.walk(exp))


def _get_args_num(net: Union[Value, Neuron, Layer, MLP]) -> int:
    if isinstance(net, Neuron):
        return len(net.parameters()) - 1
    if isinstance(net, Layer):
        return _get_args_num(net.neurons[0])
    if isinstance(net, MLP):
        return _get_args_num(net.layers[0])
    assert isinstance(net, Value)
    return 0


def _compile(net: Union[Value, Neuron, Layer, MLP]):
    args_num = _get_args_num(net)
    args_types = [ir.F32Type.get()] * args_num
    args_values = [Value(0) for _ in range(args_num)]

    @func.func(*args_types)
    def main(*args):
        net_value = net if isinstance(net, Value) else net(args_values)
        compiled_values = {v: cv for v, cv in zip(args_values, args)}
        compiler = Compiler(compiled_values)
        if isinstance(net_value, list):
            return [compiler.walk(value) for value in net_value]
        return compiler.walk(net_value)
    main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def _compile_standalone(
        net: Union[Value, Neuron, Layer, MLP]) -> ir.Module:
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            _compile(net)
        return module


def _transform(mod):
    pm = PassManager("builtin.module", context=mod.context)
    pm.add("convert-to-llvm")
    pm.run(mod.operation)
    return mod


def jit(net: Union[Value, Neuron, Layer, MLP]):
    m = _compile_standalone(net)
    execution_engine = ExecutionEngine(_transform(m))

    def jitted_net(x=None):
        c_float_p = ctypes.c_float * 1
        xs = [] if isinstance(net, Value) else x
        args = [c_float_p(v) for v in xs]
        res = c_float_p(-1.0)
        execution_engine.invoke("main", *args, res)
        return res[0]
    return jitted_net
