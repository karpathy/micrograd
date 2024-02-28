from mlir.dialects.linalg.opdsl.lang import *
from mlir.execution_engine import *
from mlir.passmanager import *
from mlir.dialects import linalg
from mlir.dialects import func
from mlir.dialects import builtin
import sys
import math
import ctypes
from micrograd.engine import Value
from micrograd.visitor import MLIRVisitor
# helps investigate segmentation faults
import faulthandler
faulthandler.enable()


def transform(mod):
    pm = PassManager("builtin.module", context=mod.context)
    pm.add("func.func(convert-linalg-to-loops)")
    pm.add("func.func(lower-affine)")
    pm.add("func.func(convert-math-to-llvm)")
    pm.add("func.func(convert-scf-to-cf)")
    pm.add("func.func(arith-expand)")
    pm.add("func.func(memref-expand)")
    pm.add("convert-vector-to-llvm")
    pm.add("finalize-memref-to-llvm")
    pm.add("convert-func-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.run(mod.operation)
    return mod


def test_basic_addition():
    a = Value(4.0)
    b = Value(2.0)
    c = a + b  # 6.
    d = a + c  # 10.
    visitor = MLIRVisitor()
    mlir_module = visitor.transform(d)
    mlir_str = str(mlir_module)
    print(mlir_str)
    # Hacky way to check we are producing any MLIR
    assert len(mlir_str) > 0
    transformed_mlir_module = transform(mlir_module)
    print(transformed_mlir_module)
    execution_engine = ExecutionEngine(transformed_mlir_module)
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.0)
    execution_engine.invoke("main", res)
    print(res[0])
    assert math.isclose(10., res[0], abs_tol=1e-08)
