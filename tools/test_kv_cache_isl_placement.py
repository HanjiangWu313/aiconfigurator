#!/usr/bin/env python3
"""
Study: KV cache transfer latency vs ISL and GPU placement.

This script answers two questions:
  1. How does KV cache transfer latency scale with ISL
     (1000, 4000, 8000, 16000, 32000)?
  2. How does GPU placement matter?
     - Intra-node (NVLink): GPU 0 → GPU 1  (within NVL8)
     - Inter-node (InfiniBand): GPU 0 → GPU 8  (cross NVL8 boundary)

Note on the AstraSimManager topology model
-------------------------------------------
AstraSim's 1-D topology treats every link identically (same bandwidth,
same latency).  To model the NVLink vs InfiniBand difference we
construct **two separate managers** with different system_spec
overrides:

  * "nvlink" manager:  bandwidth = intra_node_bw,  topology = FullyConnected
  * "ib"     manager:  bandwidth = inter_node_bw,  topology = Ring

This correctly captures latency for the two domains even though a real
system is hierarchical.  In production the manager already selects the
right tier automatically (≤ 8 GPUs → NVLink, >8 → IB), but for this
study we override manually to isolate each case.

Usage:
    python tools/test_kv_cache_isl_placement.py [--csv FILE]
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from aiconfigurator.sdk.astrasim_utils import (
    AstraSimManager,
    NETWORK_SIM_AVAILABLE,
    derive_network_params_from_system_spec,
    get_or_create_topology_config,
)

# ---------------------------------------------------------------------------
# System specs (matching the YAML files in src/aiconfigurator/systems/)
# ---------------------------------------------------------------------------

# H100 SXM — 2-tier: NVL8 (intra-node) + InfiniBand (inter-node)
H100_SXM_SYSTEM_SPEC = {
    "node": {
        "num_gpus_per_node": 8,           # NVL8 domain
        "intra_node_bw": 450_000_000_000,  # 450 GB/s NVLink per GPU
        "inter_node_bw":  25_000_000_000,  # 25 GB/s InfiniBand per GPU
        "pcie_bw":        64_000_000_000,  # 64 GB/s PCIe 5.0
        "p2p_latency":    0.000_01,        # 10 μs
    },
    "gpu": {
        "mem_bw": 3_350_000_000_000,       # 3.35 TB/s
        "mem_capacity": 85_899_345_920,    # 80 GiB
        "float16_tc_flops": 989_000_000_000_000,
    },
}

# B200 SXM — 2-tier: NVL8 (intra-node) + InfiniBand (inter-node)
B200_SXM_SYSTEM_SPEC = {
    "node": {
        "num_gpus_per_node": 8,            # NVL8 domain
        "intra_node_bw": 900_000_000_000,   # 900 GB/s NVLink 5th gen per GPU
        "inter_node_bw":  50_000_000_000,   # 50 GB/s InfiniBand (CX8) per GPU
        "pcie_bw":       128_000_000_000,   # 128 GB/s PCIe 6.0
        "p2p_latency":     0.000_01,        # 10 μs
    },
    "gpu": {
        "mem_bw": 8_000_000_000_000,        # 8 TB/s
        "mem_capacity": 193_273_528_320,    # 180 GiB
        "float16_tc_flops": 2_250_000_000_000_000,
    },
}

# GB200 SXM — 3-tier: NVL4 (intra-node) + NVSwitch/NVL72 (intra-rack) + IB (inter-rack)
GB200_SXM_SYSTEM_SPEC = {
    "node": {
        "num_gpus_per_node": 4,             # 4 GPUs per physical node
        "num_gpus_per_rack": 72,            # 72 GPUs per rack (NVL72)
        "intra_node_bw":  900_000_000_000,  # 900 GB/s NVLink 5th gen (within node)
        "inter_node_bw":  900_000_000_000,  # 900 GB/s NVLink via NVSwitch (within rack)
        "inter_rack_bw":   25_000_000_000,  # 25 GB/s InfiniBand (between racks)
        "pcie_bw":         64_000_000_000,  # 64 GB/s PCIe 5.0
        "p2p_latency":      0.000_01,       # 10 μs
        "inter_rack_latency": 0.000_005,    # 5 μs
    },
    "gpu": {
        "mem_bw": 13_400_000_000_000,       # 13.4 TB/s
        "mem_capacity": 206_158_430_208,    # 192 GiB
        "float16_tc_flops": 2_500_000_000_000_000,
    },
}

ALL_SYSTEM_SPECS = {
    "H100 SXM": H100_SXM_SYSTEM_SPEC,
    "B200 SXM": B200_SXM_SYSTEM_SPEC,
    "GB200 SXM": GB200_SXM_SYSTEM_SPEC,
}

# ---------------------------------------------------------------------------
# KV cache size model (simplified)
# ---------------------------------------------------------------------------
# KV cache size per token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
# For Llama-3.1-70B:   80 layers, 8 KV heads, 128 head_dim, FP8 = 1 byte
#   → 2 * 80 * 8 * 128 * 1 = 163,840 bytes/token ≈ 160 KB/token
# For Qwen3-32B:       64 layers, 8 KV heads, 128 head_dim, FP8 = 1 byte
#   → 2 * 64 * 8 * 128 * 1 = 131,072 bytes/token = 128 KB/token
# For Llama-3.1-405B:  126 layers, 8 KV heads, 128 head_dim, FP8 = 1 byte
#   → 2 * 126 * 8 * 128 * 1 = 258,048 bytes/token ≈ 252 KB/token

MODEL_KV_BYTES_PER_TOKEN = {
    "Llama-3.1-70B":  163_840,   # ~160 KB/token (FP8)
    "Qwen3-32B":      131_072,   # 128 KB/token  (FP8)
    "Llama-3.1-405B": 258_048,   # ~252 KB/token (FP8)
}

# Study parameters
ISL_VALUES = [1000, 4000, 8000, 16000, 32000]
PREFILL_BATCH_SIZE = 1  # single sequence for clean per-ISL measurement
MODEL_NAME = "Llama-3.1-70B"
KV_BYTES_PER_TOKEN = MODEL_KV_BYTES_PER_TOKEN[MODEL_NAME]


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: ISL sweep with auto-derived topology (realistic production path)
# ═══════════════════════════════════════════════════════════════════════════

def study_isl_sweep():
    """Sweep ISL with a single auto-derived manager (the production path).

    This uses the same code path as DisaggInferenceSession: system_spec is
    passed in, and derive_network_params_from_system_spec selects NVLink or
    IB based on total GPU count.
    """
    print("=" * 90)
    print("PART 1: KV Cache Transfer Latency vs ISL")
    print("=" * 90)
    print(f"  Model:               {MODEL_NAME}")
    print(f"  KV bytes/token:      {KV_BYTES_PER_TOKEN:,} ({KV_BYTES_PER_TOKEN/1024:.0f} KB)")
    print(f"  Prefill batch size:  {PREFILL_BATCH_SIZE}")
    print(f"  ISL values:          {ISL_VALUES}")
    print()

    # -- Demonstrate how system_spec flows --
    print("  [system_spec flow]")
    print("    system_spec['node']['intra_node_bw'] = 450 GB/s  (NVLink)")
    print("    system_spec['node']['inter_node_bw'] =  25 GB/s  (InfiniBand)")
    print("    system_spec['node']['num_gpus_per_node'] = 8      (NVL8)")
    print("    → AstraSimManager stores system_spec at construction time")
    print("    → Every simulate_* call → _build_topology(n) → get_topology_config(n)")
    print("      → derive_network_params_from_system_spec(system_spec, n)")
    print("      → if n ≤ 8: NVLink (FullyConnected, 450 GB/s)")
    print("      → if n > 8: IB     (Ring, 25 GB/s)")
    print()

    # Scenario A: intra-node (prefill GPU 0, decode GPU 1) — 2 GPUs total, NVLink
    # Scenario B: inter-node (prefill GPU 0, decode GPU 8) — 16 GPUs total, IB path
    scenarios = [
        {
            "name": "Intra-node (NVLink, FullyConnected)",
            "gpu_layout": {
                "prefill_workers": [[0]],       # GPU 0
                "decode_workers":  [[1]],        # GPU 1
            },
            "desc": "GPU 0 → GPU 1  (within single NVL8 node)",
        },
        {
            "name": "Inter-node (InfiniBand, Ring)",
            "gpu_layout": {
                "prefill_workers": [[0]],       # GPU 0
                "decode_workers":  [[8]],        # GPU 8 — crosses the NVL8 boundary
            },
            "desc": "GPU 0 → GPU 8  (across NVL8 boundary = InfiniBand)",
        },
    ]

    results = []
    manager = AstraSimManager(system_spec=H100_SXM_SYSTEM_SPEC)

    for scenario in scenarios:
        print(f"  --- {scenario['name']} ---")
        print(f"      {scenario['desc']}")

        layout = scenario["gpu_layout"]
        max_gpu = max(
            max(g for w in layout["prefill_workers"] for g in w),
            max(g for w in layout["decode_workers"] for g in w),
        )
        num_gpus = max_gpu + 1
        params = derive_network_params_from_system_spec(H100_SXM_SYSTEM_SPEC, num_gpus)
        print(f"      Total GPUs = {num_gpus}  →  BW = {params['bandwidth_gbps']:.0f} GB/s, "
              f"Topology = {params['topology']}")
        print()

        for isl in ISL_VALUES:
            kv_cache_bytes = isl * KV_BYTES_PER_TOKEN * PREFILL_BATCH_SIZE
            kv_cache_mb = kv_cache_bytes / 1e6

            t0 = time.perf_counter()
            latency_ms = manager.simulate_kv_cache_transfer(
                gpu_layout=layout,
                kv_cache_size=kv_cache_bytes,
                prefill_batch_size=PREFILL_BATCH_SIZE,
            )
            wall_us = (time.perf_counter() - t0) * 1e6

            # Analytical comparison: bytes / bandwidth
            bw_bytes_per_s = params["bandwidth_gbps"] * 1e9
            analytical_ms = (kv_cache_bytes / bw_bytes_per_s) * 1000

            row = {
                "scenario": scenario["name"],
                "isl": isl,
                "kv_cache_MB": round(kv_cache_mb, 2),
                "sim_latency_ms": round(latency_ms, 4),
                "analytical_ms": round(analytical_ms, 4),
                "bw_gbps": params["bandwidth_gbps"],
                "topology": params["topology"],
                "wall_us": round(wall_us, 1),
            }
            results.append(row)
            print(f"      ISL={isl:>6d}  KV={kv_cache_mb:>9.1f} MB  "
                  f"sim={latency_ms:>9.4f} ms  analytical={analytical_ms:>9.4f} ms  "
                  f"(sim took {wall_us:.0f} μs)")
        print()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: GPU placement study — NVLink vs InfiniBand domain in detail
# ═══════════════════════════════════════════════════════════════════════════

def study_gpu_placement():
    """Study how transfer latency changes based on GPU placement.

    In an NVL8 system (H100 SXM with 8 GPUs per node):
    - GPUs 0-7 are connected via NVLink (450 GB/s bidirectional per GPU)
    - GPUs across nodes (e.g. 0 and 8) use InfiniBand (25 GB/s per GPU)

    AstraSim creates a flat 1-D topology per simulation call.  To compare
    NVLink vs IB latency we must create managers with different bandwidth
    tiers.  The automatic path does this correctly: if total GPU count ≤ 8
    it picks NVLink bandwidth; if > 8 it picks IB.

    This section also shows direct P2P latency (simulate_p2p) to make the
    comparison crystal-clear.
    """
    print("=" * 90)
    print("PART 2: GPU Placement Impact — NVLink vs InfiniBand")
    print("=" * 90)
    print()
    print(textwrap.dedent("""\
        H100 SXM NVL8 architecture:
        ┌─────────────── Node 0 (NVLink domain) ────────────────┐
        │  GPU0 ─── GPU1 ─── GPU2 ─── GPU3                      │
        │   │        │        │        │                         │
        │  GPU4 ─── GPU5 ─── GPU6 ─── GPU7                      │
        └────────────────────────────────────────────────────────┘
                             │  InfiniBand (25 GB/s per GPU)
        ┌─────────────── Node 1 (NVLink domain) ────────────────┐
        │  GPU8 ─── GPU9 ─── GPU10 ── GPU11                     │
        │   │        │        │        │                         │
        │  GPU12 ── GPU13 ── GPU14 ── GPU15                     │
        └────────────────────────────────────────────────────────┘

        Within a node:  NVLink, 450 GB/s per GPU (FullyConnected)
        Across nodes:   InfiniBand, 25 GB/s per GPU (Ring)
        → 18× bandwidth difference!
    """))

    # Fixed KV cache sizes for comparison
    kv_sizes_mb = [10, 50, 100, 500, 1000, 2000]

    manager = AstraSimManager(system_spec=H100_SXM_SYSTEM_SPEC)

    # ── 2a: P2P latency comparison ──────────────────────────────
    print("  2a) P2P (point-to-point) transfer latency")
    print("  " + "-" * 70)

    placement_p2p = [
        # (label, src, dst, description)
        ("GPU0 → GPU1  (same node, NVLink)",       0,  1, "intra-node"),
        ("GPU0 → GPU7  (same node, NVLink)",       0,  7, "intra-node"),
        ("GPU0 → GPU8  (cross node, IB)",          0,  8, "inter-node"),
        ("GPU0 → GPU15 (cross node, IB)",          0, 15, "inter-node"),
    ]

    p2p_results = []
    for msg_mb in [1, 10, 50, 100, 500]:
        msg_bytes = int(msg_mb * 1e6)
        print(f"\n  Message size: {msg_mb} MB")
        for label, src, dst, domain in placement_p2p:
            num_gpus = max(src, dst) + 1
            latency = manager.simulate_p2p(msg_bytes, src, dst, num_gpus)
            params = derive_network_params_from_system_spec(H100_SXM_SYSTEM_SPEC, num_gpus)
            bw_gbps = params["bandwidth_gbps"]
            analytical = (msg_bytes / (bw_gbps * 1e9)) * 1000

            p2p_results.append({
                "msg_MB": msg_mb,
                "placement": label,
                "domain": domain,
                "src": src,
                "dst": dst,
                "num_gpus": num_gpus,
                "bw_gbps": bw_gbps,
                "sim_ms": round(latency, 4) if latency else None,
                "analytical_ms": round(analytical, 4),
            })
            sim_str = f"{latency:.4f}" if latency else "N/A"
            print(f"    {label:40s}  bw={bw_gbps:>6.0f} GB/s  "
                  f"sim={sim_str:>9s} ms  analytical={analytical:.4f} ms")

    # ── 2b: KV cache transfer with different placements ──────────
    print()
    print("  2b) KV cache transfer latency by placement")
    print("  " + "-" * 70)

    placement_layouts = [
        {
            "name": "Same NVL8 node: P=[GPU0], D=[GPU1]",
            "domain": "intra-node (NVLink)",
            "layout": {"prefill_workers": [[0]], "decode_workers": [[1]]},
        },
        {
            "name": "Same NVL8 node: P=[GPU0], D=[GPU4,5] (TP=2)",
            "domain": "intra-node (NVLink)",
            "layout": {"prefill_workers": [[0]], "decode_workers": [[4, 5]]},
        },
        {
            "name": "Cross NVL8 node: P=[GPU0], D=[GPU8]",
            "domain": "inter-node (IB)",
            "layout": {"prefill_workers": [[0]], "decode_workers": [[8]]},
        },
        {
            "name": "Cross NVL8 node: P=[GPU0], D=[GPU8,9] (TP=2)",
            "domain": "inter-node (IB)",
            "layout": {"prefill_workers": [[0]], "decode_workers": [[8, 9]]},
        },
        {
            "name": "Multi-worker intra: P=[GPU0],[GPU1], D=[GPU2],[GPU3]",
            "domain": "intra-node (NVLink)",
            "layout": {"prefill_workers": [[0], [1]], "decode_workers": [[2], [3]]},
        },
        {
            "name": "Multi-worker cross: P=[GPU0],[GPU1], D=[GPU8],[GPU9]",
            "domain": "inter-node (IB)",
            "layout": {"prefill_workers": [[0], [1]], "decode_workers": [[8], [9]]},
        },
    ]

    kv_results = []
    for placement in placement_layouts:
        layout = placement["layout"]
        max_gpu = max(
            max(g for w in layout["prefill_workers"] for g in w),
            max(g for w in layout["decode_workers"] for g in w),
        )
        num_gpus = max_gpu + 1
        params = derive_network_params_from_system_spec(H100_SXM_SYSTEM_SPEC, num_gpus)

        print(f"\n  {placement['name']}")
        print(f"    Domain: {placement['domain']},  "
              f"total GPUs={num_gpus},  BW={params['bandwidth_gbps']:.0f} GB/s,  "
              f"topo={params['topology']}")

        for kv_mb in kv_sizes_mb:
            kv_bytes = int(kv_mb * 1e6)
            latency = manager.simulate_kv_cache_transfer(
                gpu_layout=layout,
                kv_cache_size=kv_bytes,
                prefill_batch_size=1,
            )
            analytical = (kv_bytes / (params["bandwidth_gbps"] * 1e9)) * 1000

            kv_results.append({
                "placement": placement["name"],
                "domain": placement["domain"],
                "kv_MB": kv_mb,
                "num_gpus": num_gpus,
                "bw_gbps": params["bandwidth_gbps"],
                "topology": params["topology"],
                "sim_ms": round(latency, 4),
                "analytical_ms": round(analytical, 4),
                "speedup_vs_IB": None,  # filled in below
            })
            print(f"      KV={kv_mb:>6d} MB  sim={latency:>9.4f} ms  "
                  f"analytical={analytical:>9.4f} ms")

    return p2p_results, kv_results


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: ISL sweep with EXPLICIT NVLink vs IB comparison (side-by-side)
# ═══════════════════════════════════════════════════════════════════════════

def study_isl_nvlink_vs_ib():
    """ISL sweep with forced NVLink config vs forced IB config, side by side."""
    print("=" * 90)
    print("PART 3: ISL Sweep — NVLink vs InfiniBand Side-by-Side")
    print("=" * 90)
    print()

    # Manually create two managers: one forced NVLink, one forced IB
    nvlink_params = derive_network_params_from_system_spec(
        H100_SXM_SYSTEM_SPEC, num_gpus=2  # ≤ 8 → NVLink
    )
    ib_params = derive_network_params_from_system_spec(
        H100_SXM_SYSTEM_SPEC, num_gpus=16  # > 8 → IB
    )

    nvlink_cfg = get_or_create_topology_config(
        npus_count=16, bandwidth_gbps=nvlink_params["bandwidth_gbps"],
        latency_ns=nvlink_params["latency_ns"], topology="FullyConnected",
    )
    ib_cfg = get_or_create_topology_config(
        npus_count=16, bandwidth_gbps=ib_params["bandwidth_gbps"],
        latency_ns=ib_params["latency_ns"], topology="Ring",
    )

    nvlink_mgr = AstraSimManager(network_config=nvlink_cfg)
    ib_mgr = AstraSimManager(network_config=ib_cfg)

    print(f"  NVLink config: {nvlink_params['bandwidth_gbps']:.0f} GB/s, FullyConnected")
    print(f"  IB     config: {ib_params['bandwidth_gbps']:.0f} GB/s, Ring")
    print()

    gpu_layout = {"prefill_workers": [[0]], "decode_workers": [[1]]}

    header = (f"  {'ISL':>6s}  {'KV MB':>9s}  "
              f"{'NVLink ms':>10s}  {'IB ms':>10s}  "
              f"{'Ratio IB/NV':>11s}  {'Δ ms':>10s}")
    print(header)
    print("  " + "-" * len(header))

    comparison_results = []
    for isl in ISL_VALUES:
        kv_bytes = isl * KV_BYTES_PER_TOKEN * PREFILL_BATCH_SIZE
        kv_mb = kv_bytes / 1e6

        nv_lat = nvlink_mgr.simulate_kv_cache_transfer(gpu_layout, kv_bytes, PREFILL_BATCH_SIZE)
        ib_lat = ib_mgr.simulate_kv_cache_transfer(gpu_layout, kv_bytes, PREFILL_BATCH_SIZE)

        ratio = ib_lat / nv_lat if nv_lat > 0 else float("inf")
        delta = ib_lat - nv_lat

        comparison_results.append({
            "isl": isl,
            "kv_MB": round(kv_mb, 2),
            "nvlink_ms": round(nv_lat, 4),
            "ib_ms": round(ib_lat, 4),
            "ratio": round(ratio, 2),
            "delta_ms": round(delta, 4),
        })

        print(f"  {isl:>6d}  {kv_mb:>9.1f}  "
              f"{nv_lat:>10.4f}  {ib_lat:>10.4f}  "
              f"{ratio:>11.2f}×  {delta:>10.4f}")

    print()
    bw_ratio = nvlink_params["bandwidth_gbps"] / ib_params["bandwidth_gbps"]
    print(f"  Theoretical BW ratio: {bw_ratio:.0f}× "
          f"(NVLink {nvlink_params['bandwidth_gbps']:.0f} GB/s / "
          f"IB {ib_params['bandwidth_gbps']:.0f} GB/s)")
    print(f"  → For large transfers the IB/NVLink latency ratio should approach ~{bw_ratio:.0f}×")
    print()

    return comparison_results


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Multi-model ISL sweep
# ═══════════════════════════════════════════════════════════════════════════

def study_multi_model():
    """Show how KV cache size (and hence transfer time) varies by model."""
    print("=" * 90)
    print("PART 4: KV Cache Transfer by Model Architecture")
    print("=" * 90)
    print()

    manager = AstraSimManager(system_spec=H100_SXM_SYSTEM_SPEC)
    # Inter-node layout (the bottleneck case)
    layout = {"prefill_workers": [[0]], "decode_workers": [[8]]}

    results = []
    for model_name, kv_per_tok in MODEL_KV_BYTES_PER_TOKEN.items():
        print(f"  {model_name}: {kv_per_tok:,} bytes/token ({kv_per_tok/1024:.0f} KB)")
        for isl in ISL_VALUES:
            kv_bytes = isl * kv_per_tok
            kv_mb = kv_bytes / 1e6
            latency = manager.simulate_kv_cache_transfer(layout, kv_bytes, 1)
            results.append({
                "model": model_name,
                "kv_per_token_KB": kv_per_tok / 1024,
                "isl": isl,
                "kv_MB": round(kv_mb, 2),
                "ib_latency_ms": round(latency, 4),
            })
            print(f"    ISL={isl:>6d}  KV={kv_mb:>9.1f} MB  latency={latency:>9.4f} ms")
        print()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Cross-system comparison (H100 vs B200 vs GB200 — all tiers)
# ═══════════════════════════════════════════════════════════════════════════

def study_cross_system():
    """Compare KV cache transfer across H100, B200, GB200 at every tier.

    GB200 has a 3-tier hierarchy:
      Tier 1: intra-node   (≤4 GPUs)  → 900 GB/s NVLink
      Tier 2: intra-rack   (≤72 GPUs) → 900 GB/s NVSwitch
      Tier 3: inter-rack   (>72 GPUs) → 25 GB/s  IB

    H100 and B200 have a 2-tier hierarchy:
      Tier 1: intra-node   (≤8 GPUs)  → NVLink (450 / 900 GB/s)
      Tier 2: inter-node   (>8 GPUs)  → IB (25 / 50 GB/s)
    """
    print("=" * 90)
    print("PART 5: Cross-System Tier Comparison (H100 vs B200 vs GB200)")
    print("=" * 90)
    print()

    # Show tier derivation for each system
    print("  Tier derivation summary:")
    print("  " + "-" * 80)
    for sys_name, spec in ALL_SYSTEM_SPECS.items():
        node = spec["node"]
        gpus_per_node = node["num_gpus_per_node"]
        gpus_per_rack = node.get("num_gpus_per_rack")
        tiers_str = f"node={gpus_per_node} GPUs"
        if gpus_per_rack:
            tiers_str += f", rack={gpus_per_rack} GPUs"
        print(f"  {sys_name:12s}  {tiers_str}")

        # Show each tier
        for test_gpus, label in [(2, "few GPUs"), (gpus_per_node, "full node")]:
            p = derive_network_params_from_system_spec(spec, test_gpus)
            print(f"    {test_gpus:>4d} GPUs ({label:>12s})  → "
                  f"{p['tier']:12s}  {p['bandwidth_gbps']:>6.0f} GB/s  {p['topology']}")

        if gpus_per_rack:
            for test_gpus in [gpus_per_node + 1, gpus_per_rack]:
                p = derive_network_params_from_system_spec(spec, test_gpus)
                print(f"    {test_gpus:>4d} GPUs               → "
                      f"{p['tier']:12s}  {p['bandwidth_gbps']:>6.0f} GB/s  {p['topology']}")
            # Inter-rack
            p = derive_network_params_from_system_spec(spec, gpus_per_rack + 1)
            print(f"    {gpus_per_rack + 1:>4d} GPUs               → "
                  f"{p['tier']:12s}  {p['bandwidth_gbps']:>6.0f} GB/s  {p['topology']}")
        else:
            p = derive_network_params_from_system_spec(spec, gpus_per_node + 1)
            print(f"    {gpus_per_node + 1:>4d} GPUs               → "
                  f"{p['tier']:12s}  {p['bandwidth_gbps']:>6.0f} GB/s  {p['topology']}")
        print()

    # ── ISL sweep across all systems, at every bandwidth tier ──────
    print("  KV cache transfer latency across systems and tiers:")
    print("  " + "-" * 80)

    # Define placement scenarios per system
    # Each: (scenario_label, gpu_layout, system_name)
    cross_scenarios = []

    # H100: intra-node (NVL8), inter-node (IB)
    cross_scenarios.append(("H100 intra-node (NVL8, 450GB/s)",
        {"prefill_workers": [[0]], "decode_workers": [[1]]},
        "H100 SXM"))
    cross_scenarios.append(("H100 inter-node (IB, 25GB/s)",
        {"prefill_workers": [[0]], "decode_workers": [[8]]},
        "H100 SXM"))

    # B200: intra-node (NVL8), inter-node (IB)
    cross_scenarios.append(("B200 intra-node (NVL8, 900GB/s)",
        {"prefill_workers": [[0]], "decode_workers": [[1]]},
        "B200 SXM"))
    cross_scenarios.append(("B200 inter-node (IB, 50GB/s)",
        {"prefill_workers": [[0]], "decode_workers": [[8]]},
        "B200 SXM"))

    # GB200: intra-node (NVL4), intra-rack (NVSwitch/NVL72), inter-rack (IB)
    cross_scenarios.append(("GB200 intra-node (NVL4, 900GB/s)",
        {"prefill_workers": [[0]], "decode_workers": [[1]]},
        "GB200 SXM"))
    cross_scenarios.append(("GB200 intra-rack (NVSwitch, 900GB/s)",
        {"prefill_workers": [[0]], "decode_workers": [[4]]},   # GPU 4 is on another node but same rack
        "GB200 SXM"))
    cross_scenarios.append(("GB200 inter-rack (IB, 25GB/s)",
        {"prefill_workers": [[0]], "decode_workers": [[72]]},  # GPU 72 is on the next rack
        "GB200 SXM"))

    results = []
    for scenario_label, layout, sys_name in cross_scenarios:
        spec = ALL_SYSTEM_SPECS[sys_name]
        manager = AstraSimManager(system_spec=spec)

        max_gpu = max(
            max(g for w in layout["prefill_workers"] for g in w),
            max(g for w in layout["decode_workers"] for g in w),
        )
        num_gpus = max_gpu + 1
        params = derive_network_params_from_system_spec(spec, num_gpus)

        print(f"\n  {scenario_label}")
        print(f"    total GPUs={num_gpus}, tier={params['tier']}, "
              f"BW={params['bandwidth_gbps']:.0f} GB/s, topo={params['topology']}")

        for isl in ISL_VALUES:
            kv_bytes = isl * KV_BYTES_PER_TOKEN
            kv_mb = kv_bytes / 1e6
            latency = manager.simulate_kv_cache_transfer(layout, kv_bytes, 1)
            analytical = (kv_bytes / (params["bandwidth_gbps"] * 1e9)) * 1000

            results.append({
                "system": sys_name,
                "scenario": scenario_label,
                "tier": params["tier"],
                "bw_gbps": params["bandwidth_gbps"],
                "topology": params["topology"],
                "isl": isl,
                "kv_MB": round(kv_mb, 2),
                "sim_ms": round(latency, 4),
                "analytical_ms": round(analytical, 4),
            })
            print(f"    ISL={isl:>6d}  KV={kv_mb:>9.1f} MB  "
                  f"sim={latency:>9.4f} ms  analytical={analytical:>9.4f} ms")

    print()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Part 6: TP-paired transfer with structured GPU layouts
# ═══════════════════════════════════════════════════════════════════════════

def _build_worker_layout(num_workers: int, tp_size: int, pp_size: int, start_gpu: int) -> list[dict]:
    """Mirror of DisaggInferenceSession._build_gpu_layout helper."""
    layouts = []
    gpus_per_worker = tp_size * pp_size
    for worker_id in range(num_workers):
        worker_start = start_gpu + worker_id * gpus_per_worker
        pp_stages = []
        flat_gpu_ids = []
        for pp_rank in range(pp_size):
            stage_start = worker_start + pp_rank * tp_size
            stage_gpu_ids = list(range(stage_start, stage_start + tp_size))
            pp_stages.append(stage_gpu_ids)
            flat_gpu_ids.extend(stage_gpu_ids)
        layouts.append({
            "worker_id": worker_id,
            "gpu_ids": flat_gpu_ids,
            "pp_stages": pp_stages,
            "first_stage_gpus": pp_stages[0] if pp_stages else [],
            "last_stage_gpus": pp_stages[-1] if pp_stages else [],
            "tp_size": tp_size,
            "pp_size": pp_size,
        })
    return layouts


def _build_gpu_layout(p_tp, p_pp, p_workers, d_tp, d_pp, d_workers):
    """Build a full structured GPU layout dict (same as DisaggInferenceSession)."""
    p_gpus_per_worker = p_tp * p_pp
    d_gpus_per_worker = d_tp * d_pp

    prefill_worker_layouts = _build_worker_layout(p_workers, p_tp, p_pp, start_gpu=0)
    decode_start = p_workers * p_gpus_per_worker
    decode_worker_layouts = _build_worker_layout(d_workers, d_tp, d_pp, start_gpu=decode_start)

    # Build contiguous pairing (mirrors DisaggInferenceSession._build_gpu_layout)
    prefill_decode_pairing: dict[int, list[int]] = {}
    if p_workers >= d_workers and d_workers > 0:
        base = p_workers // d_workers
        for p_idx in range(p_workers):
            d_idx = min(p_idx // base, d_workers - 1)
            prefill_decode_pairing[p_idx] = [d_idx]
    else:
        base = d_workers // p_workers if p_workers > 0 else 1
        decode_to_prefill: dict[int, int] = {}
        for d_idx in range(d_workers):
            decode_to_prefill[d_idx] = min(d_idx // base, p_workers - 1)
        for p_idx in range(p_workers):
            prefill_decode_pairing[p_idx] = [
                d for d, p in sorted(decode_to_prefill.items()) if p == p_idx
            ]

    return {
        "prefill_workers": [w["gpu_ids"] for w in prefill_worker_layouts],
        "decode_workers": [w["gpu_ids"] for w in decode_worker_layouts],
        "prefill_worker_layouts": prefill_worker_layouts,
        "decode_worker_layouts": decode_worker_layouts,
        "prefill_decode_pairing": prefill_decode_pairing,
        "prefill_gpus_per_worker": p_gpus_per_worker,
        "decode_gpus_per_worker": d_gpus_per_worker,
        "total_prefill_gpus": p_workers * p_gpus_per_worker,
        "total_decode_gpus": d_workers * d_gpus_per_worker,
    }


def study_tp_paired_transfers():
    """Test TP-paired KV cache transfer with structured GPU layouts.

    This exercises the *new* code path in simulate_kv_cache_transfer where
    transfers go from every TP rank on the last PP stage of each prefill
    worker to every matching TP rank on the first PP stage of each decode
    worker.  We use equal prefill/decode worker counts.

    Configurations tested (all on H100 SXM):
      A) TP=1 PP=1, 2 workers each  → 4 GPUs, intra-node NVLink
      B) TP=2 PP=1, 2 workers each  → 8 GPUs, intra-node NVLink (TP-paired)
      C) TP=4 PP=1, 2 workers each  → 16 GPUs, inter-node IB (TP-paired)
      D) TP=2 PP=2, 2 workers each  → 16 GPUs, inter-node IB (PP + TP-paired)
      E) TP=4 PP=2, 1 worker each   → 16 GPUs, inter-node IB (PP + TP-paired)
      F) TP=8 PP=1, 1 worker each   → 16 GPUs, inter-node IB (max TP sharding)
    """
    print("=" * 90)
    print("PART 6: TP-Paired Transfers with Structured GPU Layouts")
    print("=" * 90)
    print()
    print("  Tests the rank-by-rank TP-paired KV cache transfer path.")
    print("  Transfers go from last_stage_gpus (prefill) → first_stage_gpus (decode).")
    print("  Equal prefill/decode workers in all cases.")
    print()

    configs = [
        # (label, p_tp, p_pp, p_workers, d_tp, d_pp, d_workers)
        ("TP=1 PP=1, 2P+2D (4 GPUs, NVLink)",    1, 1, 2,  1, 1, 2),
        ("TP=2 PP=1, 2P+2D (8 GPUs, NVLink)",    2, 1, 2,  2, 1, 2),
        ("TP=4 PP=1, 2P+2D (16 GPUs, IB)",       4, 1, 2,  4, 1, 2),
        ("TP=2 PP=2, 2P+2D (16 GPUs, IB)",       2, 2, 2,  2, 2, 2),
        ("TP=4 PP=2, 1P+1D (16 GPUs, IB)",       4, 2, 1,  4, 2, 1),
        ("TP=8 PP=1, 1P+1D (16 GPUs, IB)",       8, 1, 1,  8, 1, 1),
    ]

    manager = AstraSimManager(system_spec=H100_SXM_SYSTEM_SPEC)
    results = []

    for label, p_tp, p_pp, p_wk, d_tp, d_pp, d_wk in configs:
        layout = _build_gpu_layout(p_tp, p_pp, p_wk, d_tp, d_pp, d_wk)
        total_gpus = layout["total_prefill_gpus"] + layout["total_decode_gpus"]
        params = derive_network_params_from_system_spec(H100_SXM_SYSTEM_SPEC, total_gpus)

        print(f"  --- {label} ---")
        print(f"      Total GPUs: {total_gpus}  BW: {params['bandwidth_gbps']:.0f} GB/s  "
              f"Tier: {params['tier']}  Topo: {params['topology']}")

        # Show GPU assignment
        for i, pw in enumerate(layout["prefill_worker_layouts"]):
            print(f"      Prefill worker {i}: GPUs={pw['gpu_ids']}  "
                  f"last_stage(src)={pw['last_stage_gpus']}")
        for i, dw in enumerate(layout["decode_worker_layouts"]):
            print(f"      Decode  worker {i}: GPUs={dw['gpu_ids']}  "
                  f"first_stage(dst)={dw['first_stage_gpus']}")

        for isl in ISL_VALUES:
            kv_bytes = isl * KV_BYTES_PER_TOKEN * PREFILL_BATCH_SIZE
            kv_mb = kv_bytes / 1e6

            t0 = time.perf_counter()
            latency_ms = manager.simulate_kv_cache_transfer(
                gpu_layout=layout,
                kv_cache_size=kv_bytes,
                prefill_batch_size=PREFILL_BATCH_SIZE,
            )
            wall_us = (time.perf_counter() - t0) * 1e6

            # Analytical: total bytes / (bw * num_shards), since shards transfer in parallel
            shard_count = min(p_tp, d_tp)  # matched pairs
            kv_per_shard = kv_bytes / shard_count if shard_count > 0 else kv_bytes
            bw_bytes_per_s = params["bandwidth_gbps"] * 1e9
            analytical_ms = (kv_per_shard / bw_bytes_per_s) * 1000 if bw_bytes_per_s > 0 else 0

            row = {
                "config": label,
                "p_tp": p_tp, "p_pp": p_pp, "p_workers": p_wk,
                "d_tp": d_tp, "d_pp": d_pp, "d_workers": d_wk,
                "total_gpus": total_gpus,
                "tier": params["tier"],
                "bw_gbps": params["bandwidth_gbps"],
                "isl": isl,
                "kv_cache_MB": round(kv_mb, 2),
                "shard_count": shard_count,
                "sim_latency_ms": round(latency_ms, 4),
                "analytical_per_shard_ms": round(analytical_ms, 4),
                "wall_us": round(wall_us, 1),
            }
            results.append(row)
            print(f"      ISL={isl:>6d}  KV={kv_mb:>9.1f} MB  shards={shard_count}  "
                  f"sim={latency_ms:>9.4f} ms  analytical/shard={analytical_ms:>9.4f} ms  "
                  f"({wall_us:.0f} μs)")
        print()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Part 7: Asymmetric prefill/decode — unequal workers, TP, PP
# ═══════════════════════════════════════════════════════════════════════════

def _trace_transfers(
    manager: AstraSimManager,
    gpu_layout: dict,
    kv_cache_size: int,
    prefill_batch_size: int,
) -> list[dict]:
    """Return the exact transfer plan built by AstraSimManager."""
    return manager.build_kv_transfer_plan(
        gpu_layout=gpu_layout,
        kv_cache_size=kv_cache_size,
        prefill_batch_size=prefill_batch_size,
    )


def study_asymmetric_transfers():
    """Study KV cache transfers with asymmetric prefill/decode configs.

    Tests five categories of asymmetry:
      A) Unequal worker counts (3P+2D, 2P+4D, 1P+3D)
      B) Unequal TP (prefill/decode reshaping across different TP widths)
      C) Unequal PP (prefill PP=2, decode PP=1 → different stage GPUs)
      D) Both unequal TP and PP
      E) Many-to-few and few-to-many worker ratios with batch>1

    For each case we:
      1) Print the GPU layout (which GPUs are assigned, src/dst stage)
      2) Trace the exact transfers generated (src→dst, bytes, mode)
      3) Run AstraSim simulation and report latency
    """
    print("=" * 90)
    print("PART 7: Asymmetric Prefill/Decode — Unequal Workers, TP, PP")
    print("=" * 90)
    print()
    print("  Traces the exact transfers that simulate_kv_cache_transfer generates")
    print("  when prefill and decode have different TP, PP, or worker counts.")
    print()

    manager = AstraSimManager(
        system_spec=H100_SXM_SYSTEM_SPEC,
    )
    kv_bytes = 8000 * KV_BYTES_PER_TOKEN  # ISL=8000, ~1.3 GB
    kv_mb = kv_bytes / 1e6

    # ── Define asymmetric configs ────────────────────────────────
    # (label, p_tp, p_pp, p_workers, d_tp, d_pp, d_workers, batch_size)
    configs = [
        # Category A: Unequal worker counts, same TP/PP
        ("A1: 3P+2D, TP=1 PP=1 (5 GPUs)",                1, 1, 3,  1, 1, 2, 3),
        ("A2: 2P+4D, TP=1 PP=1 (6 GPUs)",                1, 1, 2,  1, 1, 4, 4),
        ("A3: 1P+3D, TP=2 PP=1 (8 GPUs, NVLink)",        2, 1, 1,  2, 1, 3, 2),

        # Category B: Unequal TP → overlap-based reshard
        ("B1: TP mismatch: P(TP=4) → D(TP=2), 1P+1D",   4, 1, 1,  2, 1, 1, 1),
        ("B2: TP mismatch: P(TP=2) → D(TP=4), 1P+1D",   2, 1, 1,  4, 1, 1, 1),
        ("B3: TP mismatch: P(TP=3) → D(TP=5), 1P+1D",   3, 1, 1,  5, 1, 1, 1),
        ("B4: TP mismatch: P(TP=5) → D(TP=3), 1P+1D",   5, 1, 1,  3, 1, 1, 1),
        ("B5: TP mismatch: P(TP=8) → D(TP=2), 1P+1D",   8, 1, 1,  2, 1, 1, 1),

        # Category C: Unequal PP, same TP → different stage GPU counts
        ("C1: P(TP=2,PP=2) → D(TP=2,PP=1), 1P+1D",      2, 2, 1,  2, 1, 1, 1),
        ("C2: P(TP=2,PP=1) → D(TP=2,PP=2), 1P+1D",      2, 1, 1,  2, 2, 1, 1),
        ("C3: P(TP=4,PP=2) → D(TP=4,PP=1), 2P+2D",      4, 2, 2,  4, 1, 2, 2),

        # Category D: Both unequal TP and PP
        ("D1: P(TP=4,PP=2) → D(TP=2,PP=1), 1P+1D",      4, 2, 1,  2, 1, 1, 1),
        ("D2: P(TP=2,PP=1) → D(TP=4,PP=2), 1P+1D",      2, 1, 1,  4, 2, 1, 1),

        # Category E: Unequal workers with batch>1
        ("E1: 2P+3D, TP=2 PP=1, bs=6",                   2, 1, 2,  2, 1, 3, 6),
        ("E2: 4P+1D, TP=1 PP=1, bs=4",                   1, 1, 4,  1, 1, 1, 4),
    ]

    results = []

    for label, p_tp, p_pp, p_wk, d_tp, d_pp, d_wk, bs in configs:
        layout = _build_gpu_layout(p_tp, p_pp, p_wk, d_tp, d_pp, d_wk)
        total_gpus = layout["total_prefill_gpus"] + layout["total_decode_gpus"]
        params = derive_network_params_from_system_spec(H100_SXM_SYSTEM_SPEC, total_gpus)

        print(f"  ┌── {label}")
        print(f"  │   Total GPUs: {total_gpus}  Tier: {params['tier']}  "
              f"BW: {params['bandwidth_gbps']:.0f} GB/s  batch_size={bs}")
        print(f"  │   KV total: {kv_mb:.1f} MB  KV/seq: {kv_bytes/bs/1e6:.1f} MB")
        print(f"  │")

        # Show GPU assignments
        for i, pw in enumerate(layout["prefill_worker_layouts"]):
            stages_str = " | ".join(str(s) for s in pw["pp_stages"])
            print(f"  │   Prefill[{i}]: GPUs={pw['gpu_ids']}  "
                  f"PP stages=[{stages_str}]  last_stage(src)={pw['last_stage_gpus']}")
        for i, dw in enumerate(layout["decode_worker_layouts"]):
            stages_str = " | ".join(str(s) for s in dw["pp_stages"])
            print(f"  │   Decode [{i}]: GPUs={dw['gpu_ids']}  "
                  f"PP stages=[{stages_str}]  first_stage(dst)={dw['first_stage_gpus']}")
        print(f"  │")

        # Trace transfers
        transfers = _trace_transfers(manager, layout, kv_bytes, bs)
        print(f"  │   Transfers generated: {len(transfers)}")
        for t in transfers:
            mode_tag = ""
            if t["mode"] == "tp_overlap_reshard":
                mode_tag = " [TP RESHARD]"
            print(f"  │     pp_stage={t['pp_stage']} P[{t['p_worker']}]→D[{t['d_worker']}]  "
                  f"GPU {t['src']:>2d} → GPU {t['dst']:>2d}  "
                  f"{t['bytes']/1e6:>8.1f} MB  shard={t['shard']}{mode_tag}")
        print(f"  │")

        # Run simulation
        latency_ms = manager.simulate_kv_cache_transfer(
            gpu_layout=layout,
            kv_cache_size=kv_bytes,
            prefill_batch_size=bs,
        )
        print(f"  │   AstraSim latency: {latency_ms:.4f} ms")
        print(f"  └{'─' * 75}")
        print()

        results.append({
            "config": label,
            "p_tp": p_tp, "p_pp": p_pp, "p_workers": p_wk,
            "d_tp": d_tp, "d_pp": d_pp, "d_workers": d_wk,
            "batch_size": bs,
            "total_gpus": total_gpus,
            "tier": params["tier"],
            "bw_gbps": params["bandwidth_gbps"],
            "num_transfers": len(transfers),
            "has_reshard": any(t["mode"] == "tp_overlap_reshard" for t in transfers),
            "sim_latency_ms": round(latency_ms, 4),
            "kv_total_MB": round(kv_mb, 2),
        })

    # ── Summary table ───────────────────────────────────────────
    print("  Summary Table:")
    print(f"  {'Config':<50s} {'#Xfers':>6s} {'Reshard':>8s} {'Lat(ms)':>10s}")
    print("  " + "-" * 78)
    for r in results:
        mm = "YES" if r["has_reshard"] else "no"
        print(f"  {r['config']:<50s} {r['num_transfers']:>6d} {mm:>8s} {r['sim_latency_ms']:>10.4f}")
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Study KV cache transfer cost vs ISL and GPU placement"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Save comparison results to CSV")
    args = parser.parse_args()

    if not NETWORK_SIM_AVAILABLE:
        print("ERROR: AstraSim network simulator is not available.")
        print("Make sure the library is built in network_backend/astra-network-analytical/lib/")
        sys.exit(1)

    # ── Explain system_spec flow ────────────────────────────────
    print()
    print("=" * 90)
    print("SYSTEM_SPEC DATA FLOW IN AstraSimManager")
    print("=" * 90)
    print(textwrap.dedent("""\
    
    system_spec (from h100_sxm.yaml / b200_sxm.yaml / gb200_sxm.yaml)
        │
        ├─ PerfDatabase.__init__(use_astrasim=True)
        │    └─ self._astrasim = AstraSimManager(system_spec=self.system_spec)
        │         └─ stores self._system_spec (shared by ALL simulation APIs)
        │
        └─ DisaggInferenceSession.__init__()
             ├─ Priority 1: explicit astrasim_manager parameter → use as-is
             ├─ Priority 2: network_file → AstraSimManager(network_config=file)
             └─ Priority 3: prefill_database.system_spec → AstraSimManager(system_spec=...)
    
    When any simulation API is called:
        simulate_p2p(msg, src, dst, num_gpus)
        simulate_collective(msg, num_gpus, op)    ← ALL share the SAME system_spec
        simulate_kv_cache_transfer(layout, kv, bs)
            │
            └─ _build_topology(num_gpus)
                └─ get_topology_config(num_gpus)
                    └─ derive_network_params_from_system_spec(self._system_spec, num_gpus)
                        │
                        │  2-tier systems (H100, B200):
                        │  ├─ num_gpus ≤ num_gpus_per_node → intra_node_bw, FullyConnected
                        │  └─ num_gpus > num_gpus_per_node → inter_node_bw, Ring
                        │
                        │  3-tier systems (GB200):
                        │  ├─ num_gpus ≤ num_gpus_per_node (4)  → intra_node_bw (900 GB/s), FullyConnected
                        │  ├─ num_gpus ≤ num_gpus_per_rack (72) → inter_node_bw (900 GB/s), Switch
                        │  └─ num_gpus > num_gpus_per_rack (72) → inter_rack_bw (25 GB/s),  Ring
                        │
                        └─ get_or_create_topology_config(...)
                            └─ auto_generated/<filename>.yml  (cached on disk)
    """))

    t_start = time.perf_counter()

    # Run all studies
    isl_results = study_isl_sweep()
    p2p_results, kv_placement_results = study_gpu_placement()
    comparison_results = study_isl_nvlink_vs_ib()
    model_results = study_multi_model()
    cross_results = study_cross_system()
    tp_paired_results = study_tp_paired_transfers()
    asymmetric_results = study_asymmetric_transfers()

    total_time = time.perf_counter() - t_start

    # ── Final summary ───────────────────────────────────────────
    print("=" * 90)
    print("SUMMARY & KEY TAKEAWAYS")
    print("=" * 90)
    print(textwrap.dedent(f"""\
    
    1. ISL Scaling:
       KV cache size scales linearly with ISL.  Transfer latency scales
       linearly too (dominated by bandwidth, not latency overhead).
       - ISL  1,000 → ~{ISL_VALUES[0] * KV_BYTES_PER_TOKEN / 1e6:.0f} MB KV cache
       - ISL 32,000 → ~{ISL_VALUES[-1] * KV_BYTES_PER_TOKEN / 1e6:.0f} MB KV cache
    
    2. GPU Placement — bandwidth tiers:
       H100 SXM (2-tier):  NVLink 450 GB/s  vs  IB 25 GB/s  = 18× gap
       B200 SXM (2-tier):  NVLink 900 GB/s  vs  IB 50 GB/s  = 18× gap
       GB200 SXM (3-tier): NVLink 900 GB/s  =  NVSwitch 900 GB/s  vs  IB 25 GB/s = 36× gap
       → GB200's NVL72 rack keeps 900 GB/s for up to 72 GPUs!
       → Only cross-rack (>72 GPUs) falls to 25 GB/s IB.
    
    3. Practical Impact:
       - NVLink / NVSwitch transfers: even ISL=32k finishes in ~11 ms
       - InfiniBand transfers at ISL=32k: ~195 ms (H100), ~105 ms (B200)
       - For ISL ≤ 4000 on NVLink: KV transfer < 1.5 ms (negligible)
       - Keeping prefill and decode on the same NVLink/NVSwitch domain
         eliminates the InfiniBand bottleneck entirely.
       - GB200's NVL72 lets you run 72 GPUs at NVLink speed.
    
    4. system_spec is SHARED:
       The same system_spec (and thus the same bandwidth/topology
       derivation logic including all tiers) is used by simulate_p2p,
       simulate_collective, and simulate_kv_cache_transfer.
    
    5. TP-Paired Transfers:
       With structured GPU layouts (PP stages + TP ranks), KV cache is
       sharded across TP ranks and transferred in parallel pairs:
         last_stage_gpus[i] → first_stage_gpus[i]
       This means TP=4 transfers 4 shards in parallel, each 1/4 the size.
       Congestion still applies within the same network tier.
    
    6. Asymmetric Prefill/Decode:
       - Unequal workers: sequences are round-robin'd across decode workers.
         More decode workers = each receives fewer sequences (less congestion).
       - Unequal TP (P:TP=4 -> D:TP=2): falls back to a SINGLE transfer
         of the full KV per sequence (no TP sharding benefit).
         This is a known limitation; could be improved with scatter/gather.
       - Unequal PP: only affects which GPUs are at the boundary stage,
         NOT the TP pairing. P(PP=2) sends from its last PP stage,
         D(PP=1) receives on its only stage. TP pairing works if the
         boundary TP widths match.
    
    Total study time: {total_time:.2f} s
    """))

    # ── CSV output ──────────────────────────────────────────────
    if args.csv:
        try:
            import pandas as pd
            # Save the NVLink-vs-IB comparison (most informative table)
            df = pd.DataFrame(comparison_results)
            df.to_csv(args.csv, index=False)
            print(f"  Comparison results saved to {args.csv}\n")
        except ImportError:
            # Fallback: manual CSV
            import csv
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=comparison_results[0].keys())
                writer.writeheader()
                writer.writerows(comparison_results)
            print(f"  Comparison results saved to {args.csv}\n")


if __name__ == "__main__":
    main()
