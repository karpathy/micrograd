from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
import mlir.dialects.arith as arith
import mlir.dialects.math as math
import mlir.dialects.func as func
from mlir.ir import Context, Location, InsertionPoint, Module
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir import ir
from typing import Union, Optional
import math
from ctypes import c_float, byref


class Compiler:
    """Compiler for a micrograd computation Value graph to MLIR arithmetic dialect."""

    def __init__(self, compiled_values={}):
        self.compiled_values = compiled_values

    def walk(self, value: Value) -> ir.Value:
        """Walk the Value graph and convert it an isomorphic MLIR arithmetic dialect graph."""

        if value in self.compiled_values:
            return self.compiled_values[value]
        match value._op:
            case "":
                return arith.constant(value=float(value.data), result=ir.F32Type.get())
            case "*":
                lhs, rhs = value._prev
                return arith.mulf(self.walk(lhs), self.walk(rhs))
            case "+":
                lhs, rhs = value._prev
                return arith.addf(self.walk(lhs), self.walk(rhs))
            case "ReLU":
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


def _compile_standalone(net: Union[Value, Neuron, Layer, MLP]) -> ir.Module:
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            _compile(net)
        return module


def _lower_to_llvm(mod: ir.Module) -> ir.Module:
    """Lower the MLIR module to LLVM.

    The assumption is that the module only uses standard
    dialects that can be lowered to LLVM.
    """
    pm = PassManager.parse("builtin.module(convert-to-llvm)", context=mod.context)
    pm.run(mod.operation)
    return mod


class JittedNet:
    def __init__(
        self,
        net: Union[Value, Neuron, Layer, MLP],
        m: ir.Module,
        execution_engine: ExecutionEngine,
    ):
        self.net = net
        self.m = m
        self.execution_engine = execution_engine

    def __call__(self, x: Optional[list[float]] = None):
        if isinstance(self.net, Value) and x != None:
            raise "You should not pass any arguments to a Value."
        xs = [] if isinstance(self.net, Value) else x
        args = [byref(c_float(v)) for v in xs]
        res = c_float(-1.0)
        self.execution_engine.invoke("main", *args, byref(res))
        return res.value

    def __str__(self):
        return str(self.m)


def jit(net: Union[Value, Neuron, Layer, MLP]) -> JittedNet:
    """Given a micrograd computation graph, compile it to MLIR and then to LLVM.

    You can also print the returned object to see the MLIR module.

    @return: a callable that takes the input arguments of the computation graph
    """
    m = _compile_standalone(net)
    execution_engine = ExecutionEngine(_lower_to_llvm(m))
    return JittedNet(net, m, execution_engine)
