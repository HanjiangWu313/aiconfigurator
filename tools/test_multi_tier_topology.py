#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test tiered AstraSim routing with endpoint-level NIC modeling.

This script focuses on the inter-node / inter-rack model used by
``AstraSimManager._simulate_tiered_transfers``:

* Cross-node IB traffic is modeled as a 1-D ``Switch`` topology.
* Each participating GPU contributes one NIC endpoint.
* Dense remapping sizes the generated topology to the participating endpoint
  set while preserving endpoint-sharing relationships.
* Switch links model both source-side and destination-side NIC contention.

The tests below pass even when the AstraSim Python binding is unavailable.
In that case, classification, endpoint preparation, and YAML tests still run
while the latency test is skipped.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiconfigurator.sdk.astrasim_utils import (
    NETWORK_SIM_AVAILABLE,
    AstraSimManager,
    get_or_create_topology_config,
)

H100_SXM = {
    "node": {
        "num_gpus_per_node": 8,
        "intra_node_bw": 450e9,
        "inter_node_bw": 25e9,
        "p2p_latency": 0.5e-6,
    }
}


GB200_SXM = {
    "node": {
        "num_gpus_per_node": 4,
        "num_gpus_per_rack": 72,
        "intra_node_bw": 900e9,
        "inter_node_bw": 900e9,
        "inter_rack_bw": 25e9,
        "p2p_latency": 0.5e-6,
        "inter_rack_latency": 2e-6,
    }
}


def test_classify_tier_endpoint_fabrics() -> None:
    print("=" * 72)
    print("TEST 1: Tier Classification Uses Endpoint-Level Inter-Node Fabrics")
    print("=" * 72)

    manager = AstraSimManager(system_spec=H100_SXM)

    tier_key, src_id, dst_id = manager._classify_tier(0, 3)
    assert tier_key == ("intra-node", 0), tier_key
    assert (src_id, dst_id) == (0, 3), (src_id, dst_id)

    tier_key, src_id, dst_id = manager._classify_tier(3, 8)
    assert tier_key == ("inter-node", 0), tier_key
    assert (src_id, dst_id) == (3, 8), (src_id, dst_id)

    tier_key, src_id, dst_id = manager._classify_tier(7, 12)
    assert tier_key == ("inter-node", 0), tier_key
    assert (src_id, dst_id) == (7, 12), (src_id, dst_id)

    tier_key, src_id, dst_id = manager._classify_tier(8, 0)
    assert tier_key == ("inter-node", 0), tier_key
    assert (src_id, dst_id) == (8, 0), (src_id, dst_id)

    print("  H100 2-tier classification: PASS")

    manager = AstraSimManager(system_spec=GB200_SXM)

    tier_key, src_id, dst_id = manager._classify_tier(0, 3)
    assert tier_key == ("intra-node", 0), tier_key
    assert (src_id, dst_id) == (0, 3), (src_id, dst_id)

    tier_key, src_id, dst_id = manager._classify_tier(2, 5)
    assert tier_key == ("intra-rack", 0), tier_key
    assert (src_id, dst_id) == (0, 1), (src_id, dst_id)

    tier_key, src_id, dst_id = manager._classify_tier(0, 72)
    assert tier_key == ("inter-rack", 0), tier_key
    assert (src_id, dst_id) == (0, 72), (src_id, dst_id)

    tier_key, src_id, dst_id = manager._classify_tier(4, 76)
    assert tier_key == ("inter-rack", 0), tier_key
    assert (src_id, dst_id) == (4, 76), (src_id, dst_id)

    print("  GB200 3-tier classification: PASS")

    manager = AstraSimManager(system_spec=None)
    tier_key, src_id, dst_id = manager._classify_tier(0, 8)
    assert tier_key == ("flat", 0), tier_key
    assert (src_id, dst_id) == (0, 8), (src_id, dst_id)

    print("  Flat classification: PASS")
    print()


def test_grouping_and_dense_endpoint_remap() -> None:
    print("=" * 72)
    print("TEST 2: Inter-Node Groups Densely Remap NIC Endpoint IDs")
    print("=" * 72)

    manager = AstraSimManager(system_spec=H100_SXM)
    transfers = [
        (0, 8, 100),
        (4, 8, 100),
        (4, 9, 100),
        (8, 16, 100),
        (0, 3, 100),
    ]

    tier_groups = manager._group_tiered_transfers(transfers)
    assert ("intra-node", 0) in tier_groups
    assert ("inter-node", 0) in tier_groups

    inter_group = tier_groups[("inter-node", 0)]
    assert inter_group == [
        (0, 8, 100),
        (4, 8, 100),
        (4, 9, 100),
        (8, 16, 100),
    ], inter_group

    prepared_group, num_units, participant_to_local = manager._prepare_group_for_topology("inter-node", inter_group)
    assert prepared_group == [
        (0, 2, 100),
        (1, 2, 100),
        (1, 3, 100),
        (2, 4, 100),
    ], prepared_group
    assert num_units == 5, num_units
    assert participant_to_local == {
        0: 0,
        4: 1,
        8: 2,
        9: 3,
        16: 4,
    }, participant_to_local

    print("  Inter-node dense endpoint remap: PASS")

    manager = AstraSimManager(system_spec=GB200_SXM)
    inter_rack_group = manager._group_tiered_transfers([(0, 72, 1), (4, 72, 1), (4, 76, 1)])[("inter-rack", 0)]
    prepared_group, num_units, participant_to_local = manager._prepare_group_for_topology(
        "inter-rack", inter_rack_group
    )
    assert prepared_group == [(0, 2, 1), (1, 2, 1), (1, 3, 1)], prepared_group
    assert num_units == 4, num_units
    assert participant_to_local == {
        0: 0,
        4: 1,
        72: 2,
        76: 3,
    }, participant_to_local

    print("  Inter-rack dense endpoint remap: PASS")
    print()


def test_inter_node_topology_yaml() -> None:
    print("=" * 72)
    print("TEST 3: Auto-Generated Inter-Node Input Is a 1-D Switch Topology")
    print("=" * 72)

    manager = AstraSimManager(system_spec=H100_SXM)
    params = manager._tier_topology_params("inter-node", 5)
    assert params == {
        "npus_count": 5,
        "bandwidth_gbps": 25.0,
        "latency_ns": 500.0,
        "topology": "Switch",
    }, params

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = get_or_create_topology_config(
            npus_count=params["npus_count"],
            bandwidth_gbps=params["bandwidth_gbps"],
            latency_ns=params["latency_ns"],
            topology=params["topology"],
            cache_dir=tmpdir,
        )
        content = Path(config_path).read_text()

    assert "topology: [ Switch ]" in content
    assert "npus_count: [ 5 ]" in content
    assert "bandwidth: [ 25.0 ]" in content
    assert "latency: [ 500.0 ]" in content

    print("  Inter-node YAML shape: PASS")
    print()


def test_inter_node_endpoint_contention() -> None:
    print("=" * 72)
    print("TEST 4: Endpoint-Level Inter-Node Contention")
    print("=" * 72)

    if not NETWORK_SIM_AVAILABLE:
        print("  SKIP: AstraSim Python binding not available")
        print()
        return

    msg_size = 100 * 1024 * 1024
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AstraSimManager(system_spec=H100_SXM, cache_dir=tmpdir)
        latency_single = manager._simulate_tiered_transfers([(0, 8, msg_size)])
        latency_shared_src = manager._simulate_tiered_transfers([(0, 8, msg_size), (0, 9, msg_size)])
        latency_shared_dst = manager._simulate_tiered_transfers([(0, 8, msg_size), (4, 8, msg_size)])
        latency_independent = manager._simulate_tiered_transfers([(0, 8, msg_size), (4, 9, msg_size)])

    print(f"  Single transfer:            {latency_single:.4f} ms")
    print(f"  Shared source NIC:          {latency_shared_src:.4f} ms")
    print(f"  Shared destination NIC:     {latency_shared_dst:.4f} ms")
    print(f"  Independent NIC endpoints:  {latency_independent:.4f} ms")

    assert latency_shared_src > latency_single, (latency_shared_src, latency_single)
    assert latency_shared_dst > latency_single, (latency_shared_dst, latency_single)
    assert abs(latency_independent - latency_single) < 0.01, (
        latency_independent,
        latency_single,
    )

    print("  Endpoint-level contention behavior: PASS")
    print()


def main() -> None:
    test_classify_tier_endpoint_fabrics()
    test_grouping_and_dense_endpoint_remap()
    test_inter_node_topology_yaml()
    test_inter_node_endpoint_contention()
    print("=" * 72)
    print("All multi-tier routing tests passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
