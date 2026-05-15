import sys
import pypto.language as pl
import pytest
from pypto import codegen, passes
from pypto.pypto_core import DataType, ir

def _t(name, dim=256):
    s = ir.Span.unknown()
    return ir.Var(name, ir.TensorType([ir.ConstInt(dim, DataType.INT32, s)], DataType.FP32), s)

class TestDistAllreduceCodegen:
    def test_allreduce_emits_loop(self):
        s = ir.Span.unknown()
        x, y = _t('x'), _t('y')
        c = ir.Call(ir.Op('dist.allreduce'), [x, y], s)
        b = ir.SeqStmts([ir.AssignStmt(y, c, s)], s)
        f = ir.Function('host_orch', [(x,ir.ParamDirection.In),(y,ir.ParamDirection.Out)], [], b, s, type=ir.FunctionType.Opaque, level=ir.Level.HOST, role=ir.Role.Orchestrator)
        p = ir.Program([f], 't', s)
        code = codegen.DistributedCodegen().generate(p)
        assert 'for i, ctx in enumerate(contexts):' in code
        assert 'ContinuousTensor.make' in code
        assert 'callables["allreduce_kernel"]' in code

    def test_tree_reduce_works(self):
        s = ir.Span.unknown()
        x = _t('x')
        c = ir.Call(ir.Op('dist.tree_reduce'), [x], s)
        b = ir.SeqStmts([ir.EvalStmt(c, s)], s)
        f = ir.Function('host_orch', [(x,ir.ParamDirection.In)], [], b, s, type=ir.FunctionType.Opaque, level=ir.Level.HOST, role=ir.Role.Orchestrator)
        p = ir.Program([f], 't', s)
        code = codegen.DistributedCodegen().generate(p)
        assert 'tree_reduce' in code

    def test_unknown_dist_op(self):
        s = ir.Span.unknown()
        x = _t('x')
        c = ir.Call(ir.Op('dist.unknown_op'), [x, x], s)
        b = ir.SeqStmts([ir.AssignStmt(x, c, s)], s)
        f = ir.Function('host_orch', [(x,ir.ParamDirection.In),(x,ir.ParamDirection.Out)], [], b, s, type=ir.FunctionType.Opaque, level=ir.Level.HOST, role=ir.Role.Orchestrator)
        p = ir.Program([f], 't', s)
        code = codegen.DistributedCodegen().generate(p)
        assert 'dist.unknown_op' in code

class TestGemmWithAllreduce:
    def test_gemm_allreduce_chain(self):
        @pl.program
        class Inp:
            @pl.function(type=pl.FunctionType.InCore)
            def gemm_kernel(self, a: pl.Tensor[[64,64],pl.FP32], b: pl.Tensor[[64,64],pl.FP32], c: pl.Out[pl.Tensor[[64,64],pl.FP32]]):
                with pl.incore():
                    ta=pl.load(a,[0,0],[64,64],target_memory=pl.MemorySpace.Mat)
                    tb=pl.load(b,[0,0],[64,64],target_memory=pl.MemorySpace.Mat)
                    tc=pl.matmul(pl.move(ta,target_memory=pl.MemorySpace.Left),pl.move(tb,target_memory=pl.MemorySpace.Right))
                    pl.store(tc,[0,0],c)
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def chip_orch(self, a: pl.Tensor[[64,64],pl.FP32], b: pl.Tensor[[64,64],pl.FP32], c: pl.Out[pl.Tensor[[64,64],pl.FP32]]):
                self.gemm_kernel(a,b,c)
            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, a: pl.Tensor[[64,64],pl.FP32], b: pl.Tensor[[64,64],pl.FP32], c: pl.Out[pl.Tensor[[64,64],pl.FP32]]):
                self.chip_orch(a,b,c)
        prog = passes.convert_to_ssa()(Inp)
        hf = next(f for _,f in prog.functions.items() if f.level and str(f.level)=='Level.HOST')
        ps = list(hf.params)
        s = ir.Span.unknown()
        g = ir.AssignStmt(ps[2], ir.Call(ir.GlobalVar('chip_orch'), [ps[0],ps[1],ps[2]], s), s)
        a = ir.AssignStmt(ps[2], ir.Call(ir.Op('dist.allreduce'), [ps[2],ps[2]], s), s)
        nb = ir.SeqStmts([g, a], s)
        nf = ir.Function(hf.name, [(ps[0],ir.ParamDirection.In),(ps[1],ir.ParamDirection.In),(ps[2],ir.ParamDirection.Out)], hf.return_types, nb, s, type=hf.func_type, level=hf.level, role=hf.role)
        af = [nf if f.name==hf.name else f for _,f in prog.functions.items()]
        np = ir.Program(af, prog.name, prog.span)
        code = codegen.DistributedCodegen().generate(np)
        assert 'callables["chip_orch"]' in code
        assert 'callables["allreduce_kernel"]' in code
        gp = code.find('callables["chip_orch"]')
        ap = code.find('callables["allreduce_kernel"]')
        assert gp < ap, 'allreduce must be after GEMM'

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
