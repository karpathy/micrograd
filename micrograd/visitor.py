from micrograd.engine import Value
import mlir.dialects.arith as arith
import mlir.dialects.math as math
import mlir.dialects.func as func
from mlir.ir import Context, Location, InsertionPoint, Module
from mlir import ir

class MLIRVisitor:
    
    def __init__(self):
        self.context = Context()
        self.module  = Module.create(loc = Location.unknown(context = self.context))

    def transform(self, value: Value) -> ir.Module:
        with Context(), Location.unknown():
            module  = Module.create()
            with InsertionPoint(module.body):
                @func.func()
                def main():
                    return self.walk(value)
                main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            return module
        
    def walk(self, value: Value):
        match value._op:
            case '':
                return arith.constant(value = float(value.data), result = ir.F32Type.get())
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
