# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The rank count can reach a CHIP orchestration as a runtime scalar.

Route D of the RFC #2521 O2 question carries the rank count as an orchestration
scalar instead of reading it from the comm context. Nothing in the transport
needs to be built for that: a host orchestration already binds
``pld.world_size()`` to the runner-filled ``world_size`` kwarg and passes it with
``add_scalar``, and a chip entry already unpacks scalar params with
``orch_args.scalar``. This test pins both halves end to end, so a change that
breaks either one shows up here rather than in an O2 experiment.
"""

from pathlib import Path

import pytest
import torch

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

SIZE = 64
NRANKS = 2


@pl.jit.incore
def _copy_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
) -> pl.Tensor[[1, SIZE], pl.FP32]:
    row = pl.load(inp, [0, 0], [1, SIZE])
    return pl.store(row, [0, 0], out)


@pl.jit
def _chip_pipeline(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    nblk: pl.Scalar[pl.INT64],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
) -> pl.Tensor[[1, SIZE], pl.FP32]:
    for _ in pl.range(nblk):
        out = _copy_step(inp, out)
    return out


@pl.jit.host
def _host_orch(
    inp: pl.Tensor[[NRANKS, 1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[NRANKS, 1, SIZE], pl.FP32]],
) -> pl.Tensor[[NRANKS, 1, SIZE], pl.FP32]:
    nblk = pld.world_size()
    for r in pl.range(pld.world_size()):
        _chip_pipeline(inp[r], nblk, out[r], device=r)
    return out


def test_rank_count_rides_an_orchestration_scalar():
    """``world_size`` reaches the chip entry as scalar 0, with no IR changes.

    The chip body uses the count as its loop bound, which is exactly how an L2
    orchestration would use it to derive ``B``.
    """
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    inp = torch.zeros((NRANKS, 1, SIZE), dtype=torch.float32)
    out = torch.zeros((NRANKS, 1, SIZE), dtype=torch.float32)
    compiled = _host_orch.compile(
        inp,
        out,
        config=RunConfig(
            platform="a2a3sim",
            distributed_config=DistributedConfig(device_ids=list(range(NRANKS)), num_sub_workers=0),
        ),
    )
    out_dir = Path(compiled.output_dir)

    host = next((out_dir / "orchestration").glob("*.py")).read_text()
    assert "nblk__ssa_v0 = world_size" in host, host
    assert "_ta_0.add_scalar(nblk__ssa_v0)" in host, host

    chip_path = next(out_dir.glob("next_levels/*/orchestration/*.cpp"))
    chip = chip_path.read_text()
    assert "int64_t nblk = orch_args.scalar<int64_t>(0);" in chip, chip
