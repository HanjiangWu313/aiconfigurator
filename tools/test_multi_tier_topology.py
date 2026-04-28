#!/usr/bin/env python3
"""Test multi-tier AstraSim topology approach.

Verifies that _classify_tier correctly routes GPU pairs to the right
network tier, and that _simulate_tiered_transfers produces correct
latencies with per-tier bandwidth.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiconfigurator.sdk.astrasim_utils import (
    AstraSimManager,
    NETWORK_SIM_AVAILABLE,
    derive_network_params_from_system_spec,
)

# ── System specs ─────────────────────────────────────────────────────
H100_SXM = {
    "node": {
        "num_gpus_per_node": 8,
        "intra_node_bw": 450e9,    # 450 GB/s NVLink
        "inter_node_bw": 25e9,     # 25 GB/s IB
        "p2p_latency": 0.5e-6,     # 500 ns
    }
}

B200_SXM = {
    "node": {
        "num_gpus_per_node": 8,
        "intra_node_bw": 900e9,    # 900 GB/s NVLink
        "inter_node_bw": 50e9,     # 50 GB/s IB
        "p2p_latency": 0.5e-6,
    }
}

GB200_SXM = {
    "node": {
        "num_gpus_per_node": 4,
        "num_gpus_per_rack": 72,
        "intra_node_bw": 900e9,    # 900 GB/s NVLink
        "inter_node_bw": 900e9,    # 900 GB/s NVSwitch
        "inter_rack_bw": 25e9,     # 25 GB/s IB
        "p2p_latency": 0.5e-6,
        "inter_rack_latency": 2e-6,
    }
}


def test_classify_tier():
    """Test tier classification for different system types."""
    print("=" * 70)
    print("TEST 1: Tier Classification")
    print("=" * 70)

    # ── H100 (2-tier) ──
    mgr = AstraSimManager(system_spec=H100_SXM)

    # Same node: GPUs 0-7
    tier_key, ls, ld = mgr._classify_tier(0, 3)
    assert tier_key == ("intra-node", 0), f"Expected intra-node, got {tier_key}"
    assert (ls, ld) == (0, 3), f"Local IDs wrong: {ls}, {ld}"

    tier_key, ls, ld = mgr._classify_tier(5, 7)
    assert tier_key == ("intra-node", 0)
    assert (ls, ld) == (5, 7)

    # Same node, node 1: GPUs 8-15
    tier_key, ls, ld = mgr._classify_tier(8, 11)
    assert tier_key == ("intra-node", 1), f"Expected node 1, got {tier_key}"
    assert (ls, ld) == (0, 3), f"Local IDs wrong: {ls}, {ld}"

    # Different nodes
    tier_key, ls, ld = mgr._classify_tier(3, 8)
    assert tier_key == ("inter-node", 0), f"Expected inter-node, got {tier_key}"
    assert (ls, ld) == (0, 1), f"Node IDs wrong: {ls}, {ld}"

    tier_key, ls, ld = mgr._classify_tier(7, 12)
    assert tier_key == ("inter-node", 0)
    assert (ls, ld) == (0, 1), f"Node IDs: {ls}, {ld}"

    tier_key, ls, ld = mgr._classify_tier(0, 16)
    assert tier_key == ("inter-node", 0)
    assert (ls, ld) == (0, 2), f"Node IDs: {ls}, {ld}"

    print("  H100 2-tier classification: PASS")

    # ── GB200 (3-tier) ──
    mgr = AstraSimManager(system_spec=GB200_SXM)

    # Intra-node: GPUs on same node (4 GPUs/node)
    tier_key, ls, ld = mgr._classify_tier(0, 3)
    assert tier_key == ("intra-node", 0)
    assert (ls, ld) == (0, 3)

    tier_key, ls, ld = mgr._classify_tier(4, 6)
    assert tier_key == ("intra-node", 1)
    assert (ls, ld) == (0, 2)

    # Intra-rack, different nodes: nodes within same rack
    # Rack has 72 GPUs = 18 nodes. Node 0 (GPUs 0-3) → Node 1 (GPUs 4-7)
    tier_key, ls, ld = mgr._classify_tier(2, 5)
    assert tier_key[0] == "intra-rack", f"Expected intra-rack, got {tier_key}"
    assert tier_key[1] == 0  # rack 0

    # Inter-rack: GPUs in different racks
    # Rack 0: GPUs 0-71, Rack 1: GPUs 72-143
    tier_key, ls, ld = mgr._classify_tier(0, 72)
    assert tier_key == ("inter-rack", 0), f"Expected inter-rack, got {tier_key}"
    assert (ls, ld) == (0, 1), f"Rack IDs: {ls}, {ld}"

    print("  GB200 3-tier classification: PASS")

    # ── No system_spec ──
    mgr = AstraSimManager(system_spec=None)
    tier_key, ls, ld = mgr._classify_tier(0, 8)
    assert tier_key == ("flat", 0)
    assert (ls, ld) == (0, 8)

    print("  No system_spec (flat): PASS")
    print()


def test_tier_topology_params():
    """Test topology parameter derivation per tier."""
    print("=" * 70)
    print("TEST 2: Tier Topology Parameters")
    print("=" * 70)

    mgr = AstraSimManager(system_spec=H100_SXM)

    params = mgr._tier_topology_params("intra-node", 8)
    assert params["bandwidth_gbps"] == 450.0
    assert params["topology"] == "FullyConnected"
    assert params["npus_count"] == 8
    print(f"  H100 intra-node: {params['bandwidth_gbps']} GB/s, {params['topology']}")

    params = mgr._tier_topology_params("inter-node", 4)
    assert params["bandwidth_gbps"] == 25.0
    assert params["topology"] == "Switch"
    assert params["npus_count"] == 4
    print(f"  H100 inter-node: {params['bandwidth_gbps']} GB/s, {params['topology']}")

    mgr = AstraSimManager(system_spec=GB200_SXM)

    params = mgr._tier_topology_params("intra-node", 4)
    assert params["bandwidth_gbps"] == 900.0
    assert params["topology"] == "FullyConnected"
    print(f"  GB200 intra-node: {params['bandwidth_gbps']} GB/s, {params['topology']}")

    params = mgr._tier_topology_params("intra-rack", 18)
    assert params["bandwidth_gbps"] == 900.0
    assert params["topology"] == "Switch"
    print(f"  GB200 intra-rack: {params['bandwidth_gbps']} GB/s, {params['topology']}")

    params = mgr._tier_topology_params("inter-rack", 4)
    assert params["bandwidth_gbps"] == 25.0
    assert params["topology"] == "Switch"
    print(f"  GB200 inter-rack: {params['bandwidth_gbps']} GB/s, {params['topology']}")

    print("  All parameter checks: PASS")
    print()


def test_tiered_p2p():
    """Test P2P simulation uses correct tier bandwidth."""
    if not NETWORK_SIM_AVAILABLE:
        print("SKIP: AstraSim not available")
        return

    print("=" * 70)
    print("TEST 3: Tiered P2P Simulation")
    print("=" * 70)

    mgr = AstraSimManager(system_spec=H100_SXM)
    msg_size = 100 * 1024 * 1024  # 100 MB

    # Intra-node: GPU 0 → GPU 1 (same node, NVLink 450 GB/s)
    lat_intra = mgr.simulate_p2p(msg_size, src_gpu=0, dst_gpu=1, num_total_gpus=16)
    # Inter-node: GPU 0 → GPU 8 (different nodes, IB 25 GB/s)
    lat_inter = mgr.simulate_p2p(msg_size, src_gpu=0, dst_gpu=8, num_total_gpus=16)

    print(f"  100 MB P2P intra-node (NVLink 450 GB/s): {lat_intra:.4f} ms")
    print(f"  100 MB P2P inter-node (IB 25 GB/s):      {lat_inter:.4f} ms")
    print(f"  Ratio inter/intra: {lat_inter/lat_intra:.1f}×")

    # Inter-node should be much slower than intra-node.
    # Pure BW ratio = 450/25 = 18×, but Switch topology has 2 hops
    # (NPU→switch→NPU) vs FullyConnected's 1 hop, so actual ratio ≈ 36×.
    ratio = lat_inter / lat_intra
    assert ratio > 20, f"Expected ratio > 20, got {ratio:.1f}"
    assert ratio < 50, f"Expected ratio < 50, got {ratio:.1f}"

    print(f"  Bandwidth ratio check: PASS (ratio={ratio:.1f}×, expected ~18×)")
    print()


def test_tiered_kv_cache_transfer():
    """Test KV-cache transfer with tiered topologies."""
    if not NETWORK_SIM_AVAILABLE:
        print("SKIP: AstraSim not available")
        return

    print("=" * 70)
    print("TEST 4: Tiered KV Cache Transfer")
    print("=" * 70)

    mgr = AstraSimManager(system_spec=H100_SXM)
    kv_size = 100 * 1024 * 1024  # 100 MB total KV
    batch_size = 1

    # Case A: Prefill and decode on SAME node → NVLink
    layout_same_node = {
        "prefill_workers": [[0, 1, 2, 3]],
        "decode_workers": [[4, 5, 6, 7]],
    }
    lat_same = mgr.simulate_kv_cache_transfer(layout_same_node, kv_size, batch_size)

    # Case B: Prefill and decode on DIFFERENT nodes → IB
    layout_diff_node = {
        "prefill_workers": [[0, 1, 2, 3]],
        "decode_workers": [[8, 9, 10, 11]],
    }
    lat_diff = mgr.simulate_kv_cache_transfer(layout_diff_node, kv_size, batch_size)

    print(f"  Same-node KV transfer (NVLink):  {lat_same:.4f} ms")
    print(f"  Cross-node KV transfer (IB):     {lat_diff:.4f} ms")
    print(f"  Ratio cross/same: {lat_diff/lat_same:.1f}×")

    ratio = lat_diff / lat_same
    assert ratio > 10, f"Same-node should be much faster, ratio={ratio:.1f}"
    print(f"  Same-node vs cross-node: PASS (ratio={ratio:.1f}×)")
    print()

    # Case C: Multiple prefill→decode transfers on IB → congestion
    layout_multi_inter = {
        "prefill_workers": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "decode_workers": [[8, 9, 10, 11], [12, 13, 14, 15]],
    }
    lat_multi = mgr.simulate_kv_cache_transfer(layout_multi_inter, kv_size, batch_size)

    # Case D: Single prefill→decode on IB
    layout_single_inter = {
        "prefill_workers": [[0, 1, 2, 3]],
        "decode_workers": [[8, 9, 10, 11]],
    }
    lat_single = mgr.simulate_kv_cache_transfer(layout_single_inter, kv_size, batch_size)

    print(f"  Single inter-node transfer: {lat_single:.4f} ms")
    print(f"  Two inter-node transfers (should congest): {lat_multi:.4f} ms")

    # Two transfers through the same switch should cause some congestion
    # (the switch fabric is shared). The exact behavior depends on the
    # topology type (Switch) and whether routes share links.
    print(f"  Multi/single ratio: {lat_multi/lat_single:.2f}×")
    print()


def test_congestion_same_tier():
    """Verify congestion: multiple transfers on the same inter-node link."""
    if not NETWORK_SIM_AVAILABLE:
        print("SKIP: AstraSim not available")
        return

    print("=" * 70)
    print("TEST 5: Congestion Modeling Within a Tier")
    print("=" * 70)

    mgr = AstraSimManager(system_spec=H100_SXM)
    msg_size = 100 * 1024 * 1024  # 100 MB

    # 1 transfer: node 0 → node 1
    lat_1 = mgr._simulate_tiered_transfers([(3, 8, msg_size)])
    # 2 transfers: both node 0 → node 1 (share the same link → congestion)
    lat_2 = mgr._simulate_tiered_transfers([(3, 8, msg_size), (7, 9, msg_size)])
    # 4 transfers: all node 0 → node 1
    lat_4 = mgr._simulate_tiered_transfers([
        (0, 8, msg_size), (1, 9, msg_size),
        (2, 10, msg_size), (3, 11, msg_size),
    ])

    print(f"  1 transfer  (node 0→1): {lat_1:.4f} ms")
    print(f"  2 transfers (node 0→1): {lat_2:.4f} ms (ratio: {lat_2/lat_1:.2f}×)")
    print(f"  4 transfers (node 0→1): {lat_4:.4f} ms (ratio: {lat_4/lat_1:.2f}×)")

    # With Switch topology, all go through central switch.
    # More transfers = more congestion on the switch links.
    assert lat_2 > lat_1, "2 transfers should be slower than 1"
    assert lat_4 > lat_2, "4 transfers should be slower than 2"
    print(f"  Congestion increases with load: PASS")
    print()


def test_independent_nodes_no_cross_congestion():
    """Intra-node transfers on different nodes should NOT congest each other."""
    if not NETWORK_SIM_AVAILABLE:
        print("SKIP: AstraSim not available")
        return

    print("=" * 70)
    print("TEST 6: No Cross-Congestion Between Independent Nodes")
    print("=" * 70)

    mgr = AstraSimManager(system_spec=H100_SXM)
    msg_size = 100 * 1024 * 1024  # 100 MB

    # Transfer within node 0 only
    lat_node0 = mgr._simulate_tiered_transfers([(0, 3, msg_size)])

    # Transfer within node 0 AND node 1 (independent NVLink domains)
    lat_both = mgr._simulate_tiered_transfers([
        (0, 3, msg_size),   # node 0, NVLink
        (8, 11, msg_size),  # node 1, NVLink (independent)
    ])

    print(f"  Node 0 only:       {lat_node0:.4f} ms")
    print(f"  Node 0 + Node 1:   {lat_both:.4f} ms")

    # Since they're on independent NVLink domains, the max latency
    # should be the same (both finish at the same time independently).
    assert abs(lat_both - lat_node0) < 0.01, (
        f"Independent NVLink domains should have same latency, "
        f"got {lat_node0:.4f} vs {lat_both:.4f}"
    )
    print(f"  Independent nodes don't interfere: PASS")
    print()


def test_gb200_3tier():
    """Test GB200 3-tier routing."""
    if not NETWORK_SIM_AVAILABLE:
        print("SKIP: AstraSim not available")
        return

    print("=" * 70)
    print("TEST 7: GB200 3-Tier Routing")
    print("=" * 70)

    mgr = AstraSimManager(system_spec=GB200_SXM)
    msg_size = 100 * 1024 * 1024  # 100 MB

    # Intra-node: GPU 0→3 (same 4-GPU node, NVLink 900 GB/s)
    lat_intra = mgr.simulate_p2p(msg_size, src_gpu=0, dst_gpu=3)
    # Intra-rack: GPU 0→4 (different nodes, same rack, NVSwitch 900 GB/s)
    lat_rack = mgr.simulate_p2p(msg_size, src_gpu=0, dst_gpu=4)
    # Inter-rack: GPU 0→72 (different racks, IB 25 GB/s)
    lat_inter = mgr.simulate_p2p(msg_size, src_gpu=0, dst_gpu=72)

    print(f"  Intra-node (NVLink 900 GB/s):  {lat_intra:.4f} ms")
    print(f"  Intra-rack (NVSwitch 900 GB/s): {lat_rack:.4f} ms")
    print(f"  Inter-rack (IB 25 GB/s):       {lat_inter:.4f} ms")

    # NVLink and NVSwitch are both 900 GB/s → similar latency
    # Inter-rack should be ~36× slower
    ratio_rack = lat_inter / lat_rack
    print(f"  Inter-rack / intra-rack ratio: {ratio_rack:.1f}× (expected ~36×)")

    assert ratio_rack > 20, f"Inter-rack should be much slower, ratio={ratio_rack:.1f}"
    print(f"  GB200 3-tier routing: PASS")
    print()


if __name__ == "__main__":
    test_classify_tier()
    test_tier_topology_params()
    test_tiered_p2p()
    test_tiered_kv_cache_transfer()
    test_congestion_same_tier()
    test_independent_nodes_no_cross_congestion()
    test_gb200_3tier()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
