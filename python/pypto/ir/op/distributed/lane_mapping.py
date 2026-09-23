# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Canonical lane-assignment reference for managed multi-AIV collectives (RFC #2521 K3).

RFC #2521's frozen contract item 4 makes the kernel's multi-block work split a pure
function of ``(P, B, block_idx)``: each admitted block owns a disjoint slice of peers
(``B < P``) or of one peer's payload (``B >= P``), and the per-lane completion signal
derives from the same mapping:

* ``B < P`` -- multi-peer-per-core: a block owns peers ``block_idx, block_idx + B, ...``
  and pushes each owned peer's full valid range (lane 0 of 1).
* ``B >= P`` -- peer x length: ``K = B / P`` lanes per peer; the block owns
  ``(peer = block_idx // K, lane = block_idx % K)`` and pushes the lane-th *contiguous*
  sub-range of that peer's valid payload (never interleaved by transport chunk).

Like :mod:`launch_width`, this module builds no IR and is not on any lowering or codegen
path: the actual implementation lives in the self-contained
``runtime/builtins/collectives/all_to_all_v/templates/kernel.cpp.in`` template, which is
compiled separately from these compiler sources and therefore carries its own C++ mirror
of the same formulas. These functions pin the arithmetic down once, in directly testable
form.
"""


def _check_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be int, got {type(value).__name__}")


def lanes_per_peer(p: int, b: int) -> int:
    """Lane count ``K`` per peer: ``B // P`` when ``B >= P``, else ``0``.

    ``B`` is the *admitted* block count (``cal_all_to_all_v_blocks(P, L)``), not the
    requested ``L``. The RFC's mapping guarantees ``B % P == 0`` whenever ``B >= P``;
    ``K == 0`` selects the ``B < P`` stride regime.

    Args:
        p: Rank count (the communication domain size).
        b: Admitted block count.

    Raises:
        TypeError: If ``p`` or ``b`` is not exactly ``int``.
        ValueError: If ``p`` or ``b`` is not positive.
    """
    _check_int("P", p)
    _check_int("B", b)
    if p <= 0:
        raise ValueError(f"P must be positive, got {p}")
    if b <= 0:
        raise ValueError(f"B must be positive, got {b}")
    return b // p if b >= p else 0


def lane_of_block(p: int, b: int, block_idx: int) -> tuple[int, int]:
    """Map a block index to ``(peer, lane)``.

    ``B >= P``: ``peer = block_idx // K``, ``lane = block_idx % K`` -- unique across
    ``block_idx in [0, B)``. ``B < P``: ``(block_idx, 0)``; the block also owns
    ``peer + B, peer + 2B, ...`` (see :func:`peers_of_block`).
    """
    k = lanes_per_peer(p, b)
    _check_int("block_idx", block_idx)
    if not 0 <= block_idx < b:
        raise ValueError(f"block_idx must be in [0, B), got {block_idx} (B={b})")
    if k == 0:
        return block_idx, 0
    return block_idx // k, block_idx % k


def peers_of_block(p: int, b: int, block_idx: int) -> list[int]:
    """Every peer whose work this block owns (one, or the ``B < P`` stride set)."""
    peer, _lane = lane_of_block(p, b, block_idx)
    if b >= p:
        return [peer]
    return list(range(peer, p, b))


def flat_lane_slice(valid_elems: int, k: int, lane: int) -> tuple[int, int]:
    """Contiguous ``[begin, end)`` sub-range of a peer's valid payload for ``lane``.

    ``lane_elems = ceil(valid_elems / k)``; the ``k`` ranges partition
    ``[0, valid_elems)`` exactly, and lanes beyond the payload get an empty range
    (``begin == end``), which the kernel completes without a ``TPUT``.
    """
    for name, value in (("valid_elems", valid_elems), ("k", k), ("lane", lane)):
        _check_int(name, value)
    if valid_elems < 0:
        raise ValueError(f"valid_elems must be >= 0, got {valid_elems}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not 0 <= lane < k:
        raise ValueError(f"lane must be in [0, k), got {lane} (k={k})")
    lane_elems = (valid_elems + k - 1) // k
    begin = lane * lane_elems
    begin = min(begin, valid_elems)
    end = begin + lane_elems
    end = min(end, valid_elems)
    return begin, end


__all__ = ["flat_lane_slice", "lane_of_block", "lanes_per_peer", "peers_of_block"]
