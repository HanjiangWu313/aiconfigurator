#!/usr/bin/env python3
"""Demo: How multi-tier AstraSim network simulation matters inside inference_session.

This script walks through the EXACT flow that inference_session.py uses
to compute KV-cache transfer latency, then shows a concrete case where
the network cost is *non-trivial* and changes which deployment config is
optimal.

Scenario: Llama-3.1-70B on 16× H100 SXM GPUs, disaggregated inference.
- Prefill TP=4,  Decode TP=4
- We compare different prefill/decode worker allocations:
    (a) 2P + 2D → all transfers are intra-node or cross-node
    (b) 1P + 3D → different layout
- We show how GPU placement determines NVLink vs IB and how that
  changes TTFT by *tens of milliseconds* at long ISL.

Run:
    python tools/demo_network_in_inference_session.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiconfigurator.sdk.astrasim_utils import (
    AstraSimManager,
    NETWORK_SIM_AVAILABLE,
)

# ── H100 SXM system spec (matches src/aiconfigurator/systems/h100_sxm.yaml) ──
H100_SXM_SYSTEM_SPEC = {
    "node": {
        "num_gpus_per_node": 8,
        "intra_node_bw": 450_000_000_000,   # 450 GB/s NVLink
        "inter_node_bw":  25_000_000_000,    #  25 GB/s IB
        "p2p_latency": 0.00001,              #  10 µs
    }
}


def build_gpu_layout(
    prefill_tp: int,
    prefill_pp: int,
    prefill_num_worker: int,
    decode_tp: int,
    decode_pp: int,
    decode_num_worker: int,
) -> dict:
    """
    Replicate _build_gpu_layout from inference_session.py (lines 390-435).
    
    Assigns contiguous GPU IDs:
      Prefill workers get GPUs 0..(P*tp*pp - 1)
      Decode workers  get GPUs P*tp*pp..(P*tp*pp + D*tp*pp - 1)
    """
    pfx_per = prefill_tp * prefill_pp
    dec_per = decode_tp * decode_pp

    prefill_workers = []
    for w in range(prefill_num_worker):
        start = w * pfx_per
        prefill_workers.append(list(range(start, start + pfx_per)))

    decode_start = prefill_num_worker * pfx_per
    decode_workers = []
    for w in range(decode_num_worker):
        start = decode_start + w * dec_per
        decode_workers.append(list(range(start, start + dec_per)))

    return {
        "prefill_workers": prefill_workers,
        "decode_workers": decode_workers,
        "prefill_gpus_per_worker": pfx_per,
        "decode_gpus_per_worker": dec_per,
        "total_prefill_gpus": prefill_num_worker * pfx_per,
        "total_decode_gpus": decode_num_worker * dec_per,
    }


def compute_kv_cache_size(
    isl: int,
    batch_size: int,
    hidden_size: int = 8192,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_layers: int = 80,
    dtype_bytes: int = 2,
) -> int:
    """
    Compute KV cache transfer size.
    
    In inference_session.py (line 376):
        transfer_size = num_prefill_tokens * hidden_size * dtype_size
    
    But a more precise formula (used in real workloads) is:
        per_token = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes
                    (2 for K and V)
        total = batch_size * isl * per_token
    
    For Llama-3.1-70B: 2 * 8 * 128 * 80 * 2 = 327,680 bytes/token
    """
    per_token = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes
    return batch_size * isl * per_token


def fmt_layout(layout: dict) -> str:
    """Pretty-print a layout."""
    pfx = layout["prefill_workers"]
    dec = layout["decode_workers"]
    parts = []
    for i, w in enumerate(pfx):
        parts.append(f"P{i}={w}")
    for i, w in enumerate(dec):
        parts.append(f"D{i}={w}")
    return ", ".join(parts)


def classify_all_transfers(mgr, layout, batch_size):
    """Show how each (prefill→decode) transfer gets classified."""
    prefill_workers = layout["prefill_workers"]
    decode_workers = layout["decode_workers"]
    num_pw = len(prefill_workers)
    num_dw = len(decode_workers)
    seqs_per_p = max(1, batch_size // num_pw)

    transfers = []
    for p_idx, p_gpus in enumerate(prefill_workers):
        src = p_gpus[-1]  # last GPU in pipeline
        for seq_idx in range(seqs_per_p):
            d_idx = (p_idx * seqs_per_p + seq_idx) % num_dw
            dst = decode_workers[d_idx][0]
            tier_key, local_s, local_d = mgr._classify_tier(src, dst)
            transfers.append((src, dst, tier_key[0], tier_key[1]))

    return transfers


# ========================================================================

def main():
    if not NETWORK_SIM_AVAILABLE:
        print("ERROR: AstraSim C++ library not available. Cannot run demo.")
        sys.exit(1)

    mgr = AstraSimManager(system_spec=H100_SXM_SYSTEM_SPEC)

    # ── Llama-3.1-70B parameters ──
    MODEL = "Llama-3.1-70B"
    HIDDEN = 8192
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    NUM_LAYERS = 80
    DTYPE_BYTES = 2  # fp16
    PER_TOKEN_KV = 2 * NUM_KV_HEADS * HEAD_DIM * NUM_LAYERS * DTYPE_BYTES  # 327,680 B

    print("=" * 80)
    print(f"DEMO: Network Simulation Impact on Disaggregated Inference")
    print(f"Model: {MODEL}  |  System: H100 SXM (8 GPUs/node)")
    print(f"KV cache per token: {PER_TOKEN_KV:,} bytes ({PER_TOKEN_KV/1024:.0f} KB)")
    print("=" * 80)

    # ====================================================================
    # PART 1: Trace the exact code path inside inference_session.py
    # ====================================================================
    print()
    print("─" * 80)
    print("PART 1: Code Path Walkthrough (inference_session.py → run_disagg)")
    print("─" * 80)
    print("""
    When run_disagg() is called, it executes these steps:

    1. Run prefill/decode performance models (compute-only, no network)
    2. _compute_kv_cache_transfer_size()        [line 356]
       → KV bytes = batch_size × ISL × hidden_size × dtype_size
    3. _build_gpu_layout()                       [line 390]
       → Assigns contiguous GPU IDs to prefill and decode workers
       → e.g. 2×TP4 prefill = GPUs [0-3], [4-7]
              2×TP4 decode  = GPUs [8-11], [12-15]
    4. _simulate_network_transfer()              [line 440]
       → Delegates to AstraSimManager.simulate_kv_cache_transfer()
       → NEW: Uses _classify_tier() to route each transfer to the
              correct tier (NVLink vs IB) based on GPU IDs
    5. kv_network_latency_ms is added to TTFT    [line 551]
       → disagg_summary_df['ttft'] += kv_network_latency_ms
       → This changes the effective TTFT seen by the user!
    """)

    # ====================================================================
    # PART 2: Layout comparison — where GPUs land matters
    # ====================================================================
    print("─" * 80)
    print("PART 2: GPU Layout Determines Network Tier")
    print("─" * 80)

    configs = [
        # (label, prefill_tp, prefill_pp, num_p, decode_tp, decode_pp, num_d)
        ("Config A: 2P+2D (TP=4)", 4, 1, 2, 4, 1, 2),
        ("Config B: 1P+3D (TP=4)", 4, 1, 1, 4, 1, 3),
        ("Config C: 4P+4D (TP=2)", 2, 1, 4, 2, 1, 4),
    ]

    for label, p_tp, p_pp, num_p, d_tp, d_pp, num_d in configs:
        layout = build_gpu_layout(p_tp, p_pp, num_p, d_tp, d_pp, num_d)
        total_gpus = layout["total_prefill_gpus"] + layout["total_decode_gpus"]
        print(f"\n  {label}")
        print(f"  Total GPUs: {total_gpus}")
        print(f"  Layout: {fmt_layout(layout)}")

        # Show tier classification for batch_size=1
        transfers = classify_all_transfers(mgr, layout, batch_size=1)
        for src, dst, tier_name, group_id in transfers:
            node_s, node_d = src // 8, dst // 8
            print(f"    GPU {src:2d} (node {node_s}) → GPU {dst:2d} (node {node_d})"
                  f"  →  tier={tier_name}, group={group_id}")

    # ====================================================================
    # PART 3: The main result — how network latency changes TTFT
    # ====================================================================
    print()
    print("─" * 80)
    print("PART 3: Network Latency Impact on TTFT (the punchline)")
    print("─" * 80)
    print()

    ISL_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
    BATCH_SIZE = 4

    # Config A: 2P + 2D with TP=4 → 16 GPUs
    # Prefill: GPUs [0-3], [4-7]  (node 0)
    # Decode:  GPUs [8-11], [12-15] (node 1)
    # ALL transfers are cross-node → IB 25 GB/s
    layout_cross = build_gpu_layout(4, 1, 2, 4, 1, 2)

    # Config D: 2P + 2D with TP=2 → 8 GPUs (fits on one node!)
    # Prefill: GPUs [0-1], [2-3]  (node 0)
    # Decode:  GPUs [4-5], [6-7]  (node 0)
    # ALL transfers are intra-node → NVLink 450 GB/s
    layout_same = build_gpu_layout(2, 1, 2, 2, 1, 2)

    print(f"  {'':>6s}  {'KV Cache':>10s}  {'Cross-Node (IB)':>16s}  {'Same-Node (NVL)':>16s}  "
          f"{'IB / NVL':>10s}  {'IB TTFT add':>12s}")
    print(f"  {'ISL':>6s}  {'Size (MB)':>10s}  {'Latency (ms)':>16s}  {'Latency (ms)':>16s}  "
          f"{'Ratio':>10s}  {'% if TTFT=50ms':>12s}")
    print("  " + "─" * 80)

    for isl in ISL_VALUES:
        kv_size = compute_kv_cache_size(isl, BATCH_SIZE)
        kv_mb = kv_size / 1e6

        lat_cross = mgr.simulate_kv_cache_transfer(layout_cross, kv_size, BATCH_SIZE)
        lat_same = mgr.simulate_kv_cache_transfer(layout_same, kv_size, BATCH_SIZE)

        ratio = lat_cross / lat_same if lat_same > 0 else float("inf")
        pct_of_ttft = (lat_cross / 50.0) * 100  # as % of a 50ms TTFT baseline

        print(f"  {isl:>6d}  {kv_mb:>10.1f}  {lat_cross:>16.3f}  {lat_same:>16.3f}  "
              f"{ratio:>10.1f}×  {pct_of_ttft:>11.1f}%")

    # ====================================================================
    # PART 4: Congestion — multiple prefills sending simultaneously
    # ====================================================================
    print()
    print("─" * 80)
    print("PART 4: Congestion When Multiple Prefills Send KV Simultaneously")
    print("─" * 80)
    print()

    ISL = 4096
    BATCH = 4
    kv_size = compute_kv_cache_size(ISL, BATCH)
    print(f"  ISL={ISL}, batch={BATCH}, KV={kv_size/1e6:.1f} MB per worker")
    print()

    # All cross-node: increase number of prefill workers
    for num_p in [1, 2, 4]:
        num_d = num_p  # symmetric
        layout = build_gpu_layout(4, 1, num_p, 4, 1, num_d)
        total = layout["total_prefill_gpus"] + layout["total_decode_gpus"]
        lat = mgr.simulate_kv_cache_transfer(layout, kv_size, BATCH)

        # Show transfer classification
        transfers = classify_all_transfers(mgr, layout, BATCH)
        tier_counts = {}
        for _, _, tier_name, _ in transfers:
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1

        tier_str = ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items()))
        print(f"  {num_p}P + {num_d}D ({total} GPUs): "
              f"latency = {lat:8.3f} ms  "
              f"[transfers: {tier_str}]")

    # ====================================================================
    # PART 5: The decision that changes — which config is better?
    # ====================================================================
    print()
    print("─" * 80)
    print("PART 5: Network Simulation Changes the Optimal Config")
    print("─" * 80)
    print()

    ISL = 4096
    BATCH = 4
    kv_size = compute_kv_cache_size(ISL, BATCH)

    # Simulated compute-only TTFT (from prefill model, no network)
    # These are illustrative values similar to what the analytical model produces
    COMPUTE_TTFT_TP4 = 35.0   # ms, TP=4 prefill is faster
    COMPUTE_TTFT_TP2 = 48.0   # ms, TP=2 prefill is a bit slower
    COMPUTE_TPOT_TP4 = 12.0   # ms
    COMPUTE_TPOT_TP2 = 15.0   # ms
    OSL = 512

    print(f"  Scenario: ISL={ISL}, batch={BATCH}, OSL={OSL}")
    print(f"  KV cache = {kv_size/1e6:.1f} MB")
    print()

    configs_compare = [
        # label, p_tp, p_pp, num_p, d_tp, d_pp, num_d, compute_ttft, compute_tpot
        ("2P+2D TP=4 (16 GPU, cross-node)", 4, 1, 2, 4, 1, 2,
         COMPUTE_TTFT_TP4, COMPUTE_TPOT_TP4),
        ("2P+2D TP=2 (8 GPU, same-node)",   2, 1, 2, 2, 1, 2,
         COMPUTE_TTFT_TP2, COMPUTE_TPOT_TP2),
    ]

    print(f"  {'Config':<35s}  {'Compute':>8s}  {'Network':>8s}  {'Total':>8s}  "
          f"{'E2E Latency':>12s}  {'seq/s/GPU':>10s}")
    print(f"  {'':35s}  {'TTFT':>8s}  {'KV xfer':>8s}  {'TTFT':>8s}  "
          f"{'(ms)':>12s}  {'(higher=better)':>10s}")
    print("  " + "─" * 90)

    for label, p_tp, p_pp, num_p, d_tp, d_pp, num_d, ttft_c, tpot in configs_compare:
        layout = build_gpu_layout(p_tp, p_pp, num_p, d_tp, d_pp, num_d)
        total_gpus = layout["total_prefill_gpus"] + layout["total_decode_gpus"]
        net_lat = mgr.simulate_kv_cache_transfer(layout, kv_size, BATCH)

        total_ttft = ttft_c + net_lat
        e2e = total_ttft + tpot * max(OSL - 1, 0)
        seq_s_gpu = 1000.0 / e2e / total_gpus * 1000  # approximate

        marker = ""
        print(f"  {label:<35s}  {ttft_c:>7.1f}  {net_lat:>8.3f}  {total_ttft:>7.1f}  "
              f"{e2e:>12.1f}  {seq_s_gpu:>10.4f}")

    print()
    print("  Key Insight:")
    print("  ─────────────")
    print("  The TP=4 config has BETTER compute TTFT (35ms vs 48ms), but pays")
    print("  a heavy network penalty because all KV transfers go cross-node")
    print("  over 25 GB/s IB instead of 450 GB/s NVLink.")
    print()
    print("  WITHOUT network simulation: TP=4 looks better  (lower compute TTFT)")
    print("  WITH    network simulation: the IB transfer cost can erode/reverse")
    print("  the compute advantage, especially at high ISL where KV cache is large.")
    print()
    print("  This is why the original aiconfigurator (without AstraSim) could pick")
    print("  a suboptimal config — it ignores the 10s-100s of ms of KV transfer time.")

    # ====================================================================
    # PART 6: Summary of code flow
    # ====================================================================
    print()
    print("─" * 80)
    print("PART 6: Code Flow Summary")
    print("─" * 80)
    print("""
    inference_session.py :: run_disagg()
    │
    ├─ 1. Compute TTFT/TPOT from analytical prefill/decode models
    │     (no network cost — pure GPU compute + memory bandwidth)
    │
    ├─ 2. _compute_kv_cache_transfer_size()
    │     → batch × ISL × hidden_size × dtype = KV bytes to send
    │
    ├─ 3. _build_gpu_layout()     ← contiguous GPU ID assignment
    │     → P0=[0,1,2,3], P1=[4,5,6,7], D0=[8,9,10,11], D1=[12,13,14,15]
    │
    ├─ 4. _simulate_network_transfer()
    │     → AstraSimManager.simulate_kv_cache_transfer()
    │       │
    │       ├─ Build transfers: [(src_gpu=3, dst_gpu=8, kv_per_seq), ...]
    │       │
    │       ├─ NEW: _simulate_tiered_transfers()
    │       │   ├─ _classify_tier(3, 8):
    │       │   │   node 3//8=0, node 8//8=1 → different nodes!
    │       │   │   → ("inter-node", 0), local_src=0, local_dst=1
    │       │   │
    │       │   ├─ _tier_topology_params("inter-node", 2):
    │       │   │   → Switch(2 NPUs, 25 GB/s, 10000 ns)
    │       │   │
    │       │   ├─ Create AstraSim topology + EventQueue for this tier
    │       │   ├─ Inject all chunks for this tier → shared queue = congestion!
    │       │   ├─ Run event loop → get tier latency
    │       │   │
    │       │   └─ return max(all tier latencies)
    │       │
    │       └─ Returns: kv_network_latency_ms
    │
    ├─ 5. disagg_summary_df['ttft'] += kv_network_latency_ms   ← THE KEY LINE
    │     (KV transfer happens after prefill, before decode can start)
    │
    └─ 6. Return InferenceSummary with updated TTFT, request_latency
    """)

    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
