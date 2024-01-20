from micrograd.engine import Value
from micrograd.visitor import MLIRVisitor
# helps investigate segmentation faults
import faulthandler
faulthandler.enable()


def test_basic_addition():
    a = Value(4.0)
    b = Value(2.0)
    c = a + b
    d = a + c
    visitor = MLIRVisitor()
    mlir_str = str(visitor.transform(d))
    print(mlir_str)
    # Hacky way to check we are producing any MLIR
    assert len(mlir_str) > 0