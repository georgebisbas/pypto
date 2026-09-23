# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the RFC #2521 K3 lane-assignment reference (``lane_mapping``)."""

import sys

import pytest
from pypto.ir.op.distributed.lane_mapping import (
    flat_lane_slice,
    lane_of_block,
    lanes_per_peer,
    peers_of_block,
)
from pypto.ir.op.distributed.launch_width import cal_all_to_all_v_blocks

# The RFC's own EP8 worked table (plan 112) plus the K2 campaign's EP4 points.
WORKED_POINTS = [
    (8, 1),
    (8, 7),
    (8, 8),
    (8, 10),
    (8, 15),
    (8, 16),
    (16, 7),
    (16, 16),
    (4, 2),
    (4, 10),
]


@pytest.mark.parametrize(("p", "req_l"), WORKED_POINTS)
def test_lane_mapping_is_a_bijection_or_a_partition(p, req_l):
    """Every admitted block maps in-range; B >= P is a peer x lane bijection."""
    b = cal_all_to_all_v_blocks(p, req_l)
    k = lanes_per_peer(p, b)

    seen = set()
    for idx in range(b):
        peer, lane = lane_of_block(p, b, idx)
        assert 0 <= peer < p, f"(P={p}, B={b}, idx={idx}) -> peer {peer} out of range"
        if k == 0:
            assert lane == 0
        else:
            assert 0 <= lane < k
            assert (peer, lane) not in seen, f"collision at block_idx={idx}"
            seen.add((peer, lane))

    if k > 0:
        assert seen == {(peer, lane) for peer in range(p) for lane in range(k)}
    else:
        owned = [q for idx in range(b) for q in peers_of_block(p, b, idx)]
        assert sorted(owned) == list(range(p)), "stride sets must partition the peers"


def test_stride_regime_owns_every_bth_peer():
    """B < P: block ``idx`` owns ``idx, idx + B, ...`` and lane 0 only."""
    assert lanes_per_peer(4, 2) == 0
    assert lane_of_block(4, 2, 0) == (0, 0)
    assert peers_of_block(4, 2, 0) == [0, 2]
    assert peers_of_block(4, 2, 1) == [1, 3]


def test_peer_length_regime_mapping():
    """B >= P: K = B / P; peer = idx // K, lane = idx % K."""
    assert lanes_per_peer(4, 8) == 2
    assert peers_of_block(4, 8, 5) == [2]
    assert lane_of_block(4, 8, 5) == (2, 1)
    assert lane_of_block(8, 16, 15) == (7, 1)
    assert lane_of_block(8, 8, 0) == (0, 0)  # K == 1 degenerates to one block per peer


@pytest.mark.parametrize(("valid_elems", "k"), [(0, 2), (1, 1), (2, 4), (10, 3), (9, 3), (7, 4), (100, 7)])
def test_flat_lane_slice_partitions_the_valid_range(valid_elems, k):
    """Slices are contiguous, ordered, disjoint, and cover ``[0, valid_elems)``."""
    parts = [flat_lane_slice(valid_elems, k, lane) for lane in range(k)]
    assert parts[0][0] == 0
    for (begin, end), (next_begin, _next_end) in zip(parts, parts[1:]):
        assert begin <= end <= valid_elems
        assert end == next_begin, "slices must be adjacent, not interleaved"
    assert parts[-1][1] == valid_elems


def test_flat_lane_slice_examples():
    """Ceil-division split; empty tail lanes allowed (``valid_elems < K``)."""
    assert [flat_lane_slice(10, 3, lane) for lane in range(3)] == [(0, 4), (4, 8), (8, 10)]
    assert [flat_lane_slice(2, 4, lane) for lane in range(4)] == [(0, 1), (1, 2), (2, 2), (2, 2)]


@pytest.mark.parametrize(("p", "b"), [(0, 4), (-1, 4), (4, 0), (4, -1)])
def test_rejects_non_positive_p_b(p, b):
    with pytest.raises(ValueError, match="must be positive"):
        lanes_per_peer(p, b)


@pytest.mark.parametrize(("p", "b", "idx"), [(4, 8, -1), (4, 8, 8), (4, 8, 9)])
def test_rejects_out_of_range_block_idx(p, b, idx):
    with pytest.raises(ValueError, match="block_idx must be in"):
        lane_of_block(p, b, idx)


@pytest.mark.parametrize(
    ("valid_elems", "k", "lane", "match"),
    [(5, 0, 0, "k must be positive"), (5, 3, 3, "lane must be in"), (-1, 2, 0, "valid_elems")],
)
def test_flat_lane_slice_rejects_bad_input(valid_elems, k, lane, match):
    with pytest.raises(ValueError, match=match):
        flat_lane_slice(valid_elems, k, lane)


def test_rejects_non_integer_types():
    with pytest.raises(TypeError, match="must be int"):
        lanes_per_peer(2.5, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be int"):
        flat_lane_slice(8, True, 0)  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
