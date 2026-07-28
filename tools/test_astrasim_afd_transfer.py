#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small standalone checks for AstraSim AFD fanout transfer modeling.

This intentionally lives in tools/ instead of the unit-test tree.  It exercises
the multi-dimensional AFD path with a tiny fake topology so the check is fast
and does not depend on the AstraSim native library being present.
"""

from __future__ import annotations

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from aiconfigurator.sdk import astrasim_utils
from aiconfigurator.sdk.astrasim_utils import AstraSimManager


class ConstantDelayTopology:
    def __init__(self, delay_ns: int):
        self.delay_ns = delay_ns
        self.calls: list[tuple[int, int, int]] = []

    def send(self, src_gpu: int, dst_gpu: int, size_bytes: int) -> int:
        self.calls.append((src_gpu, dst_gpu, size_bytes))
        return self.delay_ns


def _manager_with_topology(topology: ConstantDelayTopology) -> AstraSimManager:
    manager = AstraSimManager()
    manager._build_multidim_topology = lambda num_gpus: (topology, "fake_multidim.yml")
    return manager


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def test_one_sender_fanout_serializes_sender() -> None:
    topology = ConstantDelayTopology(delay_ns=1_000_000)
    manager = _manager_with_topology(topology)

    latency_ms = manager.simulate_multidim_afd(
        attn_gpu_ids=[0],
        ffn_gpu_ids=[1, 2, 3, 4],
        sender_bytes_per_gpu=400,
        receiver_bytes_per_gpu=100,
        pre_dispatch=True,
    )

    _assert_close("one sender fanout", latency_ms, 4.0)
    assert topology.calls == [
        (0, 1, 100),
        (0, 2, 100),
        (0, 3, 100),
        (0, 4, 100),
    ]


def test_many_senders_contend_on_one_receiver() -> None:
    topology = ConstantDelayTopology(delay_ns=1_000_000)
    manager = _manager_with_topology(topology)

    latency_ms = manager.simulate_multidim_afd(
        attn_gpu_ids=[0, 1, 2, 3],
        ffn_gpu_ids=[4],
        sender_bytes_per_gpu=100,
        receiver_bytes_per_gpu=400,
        pre_dispatch=True,
    )

    _assert_close("many senders to one receiver", latency_ms, 4.0)
    assert topology.calls == [
        (0, 4, 100),
        (1, 4, 100),
        (2, 4, 100),
        (3, 4, 100),
    ]


def test_four_by_four_is_endpoint_not_global_serialized() -> None:
    topology = ConstantDelayTopology(delay_ns=1_000_000)
    manager = _manager_with_topology(topology)

    latency_ms = manager.simulate_multidim_afd(
        attn_gpu_ids=[0, 1, 2, 3],
        ffn_gpu_ids=[4, 5, 6, 7],
        sender_bytes_per_gpu=400,
        receiver_bytes_per_gpu=400,
        pre_dispatch=True,
    )

    _assert_close("4x4 endpoint makespan", latency_ms, 4.0)
    assert len(topology.calls) == 16


def main() -> None:
    old_available = astrasim_utils.NETWORK_SIM_UNAWARE_AVAILABLE
    try:
        astrasim_utils.NETWORK_SIM_UNAWARE_AVAILABLE = True
        test_one_sender_fanout_serializes_sender()
        test_many_senders_contend_on_one_receiver()
        test_four_by_four_is_endpoint_not_global_serialized()
    finally:
        astrasim_utils.NETWORK_SIM_UNAWARE_AVAILABLE = old_available

    print("AstraSim AFD transfer checks passed.")


if __name__ == "__main__":
    main()
