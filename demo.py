import pypto.language as pl
from pypto.ir import compile, OptimizationStrategy
from pypto.backend import BackendType

@pl.program
class VectorAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self,
             a: pl.Tensor[[128, 128], pl.FP32],
             b: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_c = pl.add(tile_a, tile_b)
        return pl.store(tile_c, [0, 0], a)

# Compile — produces output files in build_output/
compiled = compile(
    VectorAdd,
    output_dir="/tmp/pypto_output",
    strategy=OptimizationStrategy.Default,
    dump_passes=True,
    backend_type=BackendType.Ascend910B,
)

# Print the output directory
print(f"Output: {compiled.output_dir}")
