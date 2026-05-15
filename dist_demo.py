from pypto import codegen, passes
import pypto.language as pl

@pl.program
class Input:
    @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
    def pod_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

program = passes.convert_to_ssa()(Input)
program = passes.outline_hierarchy_scopes()(program)

dist = codegen.DistributedCodegen()
cpp_source = dist.generate(program)

with open("distributed_main.cpp", "w") as f:
    f.write(cpp_source)

print("generated", len(cpp_source), "chars")