"""This is a small JIT compiler for micrograd computation graphs using MLIR.

The MLIR is lowered to LLVM IR and then executed using an LLVM JIT engine.
The comments in the file are meant to be liberal as this is a demonstration
and learning project.
"""

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
from ctypes import c_float, byref, pointer


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


def _get_results_num(net: Union[Value, Neuron, Layer, MLP]) -> int:
    if isinstance(net, Layer):
        return len(net.neurons)
    if isinstance(net, MLP):
        return _get_results_num(net.layers[-1])
    assert isinstance(net, Value) or isinstance(net, Neuron)
    return 1


def _compile(net: Union[Value, Neuron, Layer, MLP]):
    """Adds the main method to a MLIR module.

    This function assumes it is called within a context and insertion point.
    """
    args_num = _get_args_num(net)
    args_types = [ir.F32Type.get()] * args_num
    args_values = [Value(0) for _ in range(args_num)]

    @func.func(*args_types)
    def main(*args):
        # This is a bit of a hack to figure out the computation graph.
        # Rather than model the various remaining types such as
        # Neuron, Layer, and MLP, we instead execute the computation
        # and since the result is a Value it encodes the whole graph.
        # This is OK since the point of JIT is to speedup subsequent
        # executions.
        net_value = net if isinstance(net, Value) else net(args_values)
        # The computation graph earlier was created with seed values of Value(0).
        # We now need to replace these with the actual arguments provided to the
        # MLIR main function.
        # We accomplish this by creating a mapping from the seed values to the
        # compiled arguments (cv). The walk method will replace the seed values
        # when traversing the graph wth the actual arguments
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

        num_results = _get_results_num(self.net)
        FloatResultArrayType = c_float * num_results
        res = FloatResultArrayType(-1)

        # ExecutionEngine has odd semantics if an argument is a pointer.
        # Some networks can return a single value, others a list.
        # This also changes the type of MLIR that is lowered to LLVM such that the
        # return value must be in argument to the function now.
        # https://github.com/llvm/llvm-project/issues/83599
        if num_results == 1:
            args = args + [byref(res)]
        else:
            args = [pointer(pointer(res))] + args

        self.execution_engine.invoke("main", *args)
        return res[0] if num_results == 1 else [res[i] for i in range(num_results)]

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
