#!/usr/bin/env python3
"""
Test script to measure KV cache transfer overhead across disaggregated configurations.

Sweeps model, ISL, batch size, TP parallelism, worker counts, and network topologies
to show how much of TTFT and request latency is attributable to KV cache transfer.
"""

import copy
import os
import sys
import time

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.inference_session import DisaggInferenceSession
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path

# ── Network topology configs ────────────────────────────────────
_NET_INPUT_DIR = os.path.join(
    _PROJECT_ROOT, "network_backend", "astra-network-analytical", "input"
)
NETWORK_TOPOLOGIES = {
    "Ring": os.path.join(_NET_INPUT_DIR, "Ring.yml"),
    "FullyConnected": os.path.join(_NET_INPUT_DIR, "FullyConnected.yml"),
    "Switch": os.path.join(_NET_INPUT_DIR, "Switch.yml"),
}

# ── Disagg configurations to sweep ─────────────────────────────
# Each entry: (label, model_path, system, version, isl, osl,
#              prefill_tp, decode_tp, prefill_bs, decode_bs,
#              prefill_workers, decode_workers, quant_mode)

CONFIGS = [
    # --- Vary ISL (sequence length drives KV cache size) ---
    ("Qwen3-32B ISL=1024",  "Qwen/Qwen3-32B",  "h200_sxm", "1.2.0rc5", 1024, 256, 1, 2, 1, 32, 2, 2, "fp8_block"),
    ("Qwen3-32B ISL=2048",  "Qwen/Qwen3-32B",  "h200_sxm", "1.2.0rc5", 2048, 256, 1, 2, 1, 32, 2, 2, "fp8_block"),
    ("Qwen3-32B ISL=4096",  "Qwen/Qwen3-32B",  "h200_sxm", "1.2.0rc5", 4096, 512, 1, 2, 1, 56, 4, 2, "fp8_block"),
    ("Qwen3-32B ISL=8192",  "Qwen/Qwen3-32B",  "h200_sxm", "1.2.0rc5", 8192, 512, 1, 2, 1, 56, 4, 2, "fp8_block"),

    # --- Vary prefill batch size (more sequences → bigger KV transfer) ---
    ("Qwen3-32B pBS=1",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 4, 2, "fp8_block"),
    ("Qwen3-32B pBS=2",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 2, 56, 4, 2, "fp8_block"),
    ("Qwen3-32B pBS=4",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 4, 56, 4, 2, "fp8_block"),

    # --- Vary worker counts (more workers → more parallel KV transfers) ---
    # Note: Ring.yml has 8 NPUs, so total GPUs (p_workers*p_tp + d_workers*d_tp) must be ≤ 8
    ("Qwen3-32B 1p:1d",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 1, 1, "fp8_block"),
    ("Qwen3-32B 2p:2d",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 2, 2, "fp8_block"),
    ("Qwen3-32B 4p:2d",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 4, 2, "fp8_block"),
    ("Qwen3-32B 2p:3d",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 2, 3, "fp8_block"),

    # --- Vary TP size (larger TP → fewer bytes per shard but different GPU layout) ---
    # Note: total GPUs = p_workers*p_tp + d_workers*d_tp must be ≤ 8 (Ring.yml)
    ("Qwen3-32B dTP=1",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 1, 1, 56, 4, 2, "fp8_block"),
    ("Qwen3-32B dTP=2",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 4, 2, "fp8_block"),
    ("Qwen3-32B dTP=4",  "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 2, 4, 1, 56, 1, 1, "fp8_block"),
    ("Qwen3-32B reshape 2->4", "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 2, 4, 1, 56, 1, 1, "fp8_block"),
    ("Qwen3-32B reshape 4->2", "Qwen/Qwen3-32B", "h200_sxm", "1.2.0rc5", 4000, 500, 4, 2, 1, 56, 1, 1, "fp8_block"),

    # --- Different model (MoE vs Dense) ---
    ("Qwen3-30B-A3B ISL=4k", "Qwen/Qwen3-30B-A3B", "h200_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 4, 2, "fp8_block"),

    # --- Different system ---
    ("Qwen3-32B H100",  "Qwen/Qwen3-32B", "h100_sxm", "1.2.0rc5", 4000, 500, 1, 2, 1, 56, 4, 2, "fp8"),
]

QUANT_PRESETS = {
    "fp8_block": dict(
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        comm_quant_mode=common.CommQuantMode.half,
    ),
    "fp8": dict(
        gemm_quant_mode=common.GEMMQuantMode.fp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8,
        comm_quant_mode=common.CommQuantMode.half,
    ),
}


def _make_model_config(tp: int, quant: str) -> config.ModelConfig:
    """Create a ModelConfig with given TP and quant preset."""
    return config.ModelConfig(
        tp_size=tp,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        **QUANT_PRESETS[quant],
    )


def run_single_config(
    label: str,
    model_path: str,
    system: str,
    version: str,
    isl: int,
    osl: int,
    prefill_tp: int,
    decode_tp: int,
    prefill_bs: int,
    decode_bs: int,
    prefill_workers: int,
    decode_workers: int,
    quant: str,
    network_file: str | None,
    topology_name: str,
    gpu_layout_strategy: str,
) -> dict:
    """Run a single disagg configuration and return a result dict."""
    systems_dir = get_system_config_path()
    backend = TRTLLMBackend()
    database = PerfDatabase(system, "trtllm", version, systems_dir=str(systems_dir))

    runtime_cfg = config.RuntimeConfig(
        batch_size=prefill_bs,
        beam_width=1,
        isl=isl,
        osl=osl,
        prefix=0,
    )
    prefill_mc = _make_model_config(prefill_tp, quant)
    decode_mc = _make_model_config(decode_tp, quant)

    session = DisaggInferenceSession(
        prefill_database=database,
        prefill_backend=backend,
        decode_database=copy.deepcopy(database),
        decode_backend=backend,
        network_file=network_file,
        gpu_layout_strategy=gpu_layout_strategy,
    )

    t0 = time.perf_counter()
    summary = session.run_disagg(
        model_path=model_path,
        runtime_config=runtime_cfg,
        prefill_model_config=prefill_mc,
        prefill_batch_size=prefill_bs,
        prefill_num_worker=prefill_workers,
        decode_model_config=decode_mc,
        decode_batch_size=decode_bs,
        decode_num_worker=decode_workers,
    )
    wall_time = (time.perf_counter() - t0) * 1000  # ms

    df = summary.get_summary_df()
    row = df.iloc[0]

    net_info = summary.get_network_info()
    net_lat = net_info["kv_network_latency_ms"]
    kv_bytes = net_info["kv_cache_size_bytes"]
    kv_pct = summary.get_kv_cache_transfer_pct()

    ttft = float(row["ttft"])
    tpot = float(row["tpot"])
    req_lat = float(row["request_latency"])
    tokens_s_gpu = float(row["tokens/s/gpu"])
    num_gpus = int(row["num_total_gpus"])

    return {
        "label": label,
        "model": model_path.split("/")[-1],
        "system": system,
        "topology": topology_name,
        "gpu_layout_strategy": gpu_layout_strategy,
        "isl": isl,
        "osl": osl,
        "p_tp": prefill_tp,
        "d_tp": decode_tp,
        "p_bs": prefill_bs,
        "d_bs": decode_bs,
        "p_workers": prefill_workers,
        "d_workers": decode_workers,
        "num_gpus": num_gpus,
        "ttft_ms": round(ttft, 3),
        "tpot_ms": round(tpot, 3),
        "request_lat_ms": round(req_lat, 3),
        "kv_cache_MB": round(kv_bytes / 1e6, 2),
        "net_latency_ms": round(net_lat, 3),
        "kv_pct_of_ttft": round(kv_pct, 2),
        "kv_pct_of_e2e": round(net_lat / req_lat * 100, 2) if req_lat > 0 and net_lat > 0 else 0.0,
        "tokens/s/gpu": round(tokens_s_gpu, 2),
        "sim_wall_ms": round(wall_time, 1),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure KV cache transfer overhead across disagg configurations"
    )
    parser.add_argument(
        "--topology",
        choices=list(NETWORK_TOPOLOGIES.keys()) + ["all", "none"],
        default="Ring",
        help="Network topology to simulate (default: Ring). "
             "Use 'all' to sweep topologies, 'none' for baseline without AstraSim.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV file")
    parser.add_argument(
        "--layout-strategy",
        choices=["segregated_by_phase", "paired_prefill_decode_per_node"],
        default="segregated_by_phase",
        help="How logical prefill/decode workers are mapped onto GPU IDs.",
    )
    args = parser.parse_args()

    # Resolve topology list
    if args.topology == "all":
        topo_list = list(NETWORK_TOPOLOGIES.items())
    elif args.topology == "none":
        topo_list = [("none", None)]
    else:
        topo_list = [(args.topology, NETWORK_TOPOLOGIES[args.topology])]

    print("=" * 100)
    print("KV Cache Transfer Overhead Study")
    print("=" * 100)
    print(f"Topologies: {[t[0] for t in topo_list]}")
    print(f"Layout strategy: {args.layout_strategy}")
    print(f"Configurations: {len(CONFIGS)}")
    print()

    results = []
    total = len(CONFIGS) * len(topo_list)
    for i, (topo_name, net_file) in enumerate(topo_list):
        for j, cfg in enumerate(CONFIGS):
            idx = i * len(CONFIGS) + j + 1
            label = cfg[0]
            print(f"[{idx:3d}/{total}] {topo_name:16s} | {label:30s} ... ", end="", flush=True)
            try:
                result = run_single_config(
                    label=label,
                    model_path=cfg[1],
                    system=cfg[2],
                    version=cfg[3],
                    isl=cfg[4],
                    osl=cfg[5],
                    prefill_tp=cfg[6],
                    decode_tp=cfg[7],
                    prefill_bs=cfg[8],
                    decode_bs=cfg[9],
                    prefill_workers=cfg[10],
                    decode_workers=cfg[11],
                    quant=cfg[12],
                    network_file=net_file,
                    topology_name=topo_name,
                    gpu_layout_strategy=args.layout_strategy,
                )
                results.append(result)
                net_ms = result["net_latency_ms"]
                kv_mb = result["kv_cache_MB"]
                kv_pct = result["kv_pct_of_ttft"]
                print(
                    f"OK  KV={kv_mb:>8.1f} MB  net={net_ms:>7.3f} ms  "
                    f"kv/ttft={kv_pct:>5.1f}%  ttft={result['ttft_ms']:.1f} ms  "
                    f"({result['sim_wall_ms']:.0f} ms wall)"
                )
            except Exception as e:
                print(f"FAIL: {e}")
                continue

    if not results:
        print("\nNo successful runs.")
        return

    # ── Summary table ───────────────────────────────────────────
    df = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    display_cols = [
        "label", "topology", "system", "isl", "osl",
        "p_tp", "d_tp", "p_bs", "d_bs", "p_workers", "d_workers",
        "num_gpus", "ttft_ms", "tpot_ms", "request_lat_ms",
        "kv_cache_MB", "net_latency_ms", "kv_pct_of_ttft", "kv_pct_of_e2e",
        "tokens/s/gpu",
    ]
    print(df[display_cols].to_string(index=False))

    # ── Key observations ────────────────────────────────────────
    print("\n" + "-" * 100)
    print("KEY OBSERVATIONS")
    print("-" * 100)

    with_net = df[df["net_latency_ms"] > 0]
    if len(with_net) > 0:
        print(f"  Configs with AstraSim network latency: {len(with_net)}/{len(df)}")
        print(f"  KV cache size range:      {with_net['kv_cache_MB'].min():.1f} – {with_net['kv_cache_MB'].max():.1f} MB")
        print(f"  Network latency range:    {with_net['net_latency_ms'].min():.3f} – {with_net['net_latency_ms'].max():.3f} ms")
        print(f"  KV % of TTFT range:       {with_net['kv_pct_of_ttft'].min():.1f}% – {with_net['kv_pct_of_ttft'].max():.1f}%")
        print(f"  KV % of E2E latency range:{with_net['kv_pct_of_e2e'].min():.1f}% – {with_net['kv_pct_of_e2e'].max():.1f}%")
        print(f"  Mean KV % of TTFT:        {with_net['kv_pct_of_ttft'].mean():.1f}%")
        print(f"  Mean KV % of E2E:         {with_net['kv_pct_of_e2e'].mean():.1f}%")

        # Group by ISL to show scaling
        isl_groups = with_net.groupby("isl").agg(
            kv_cache_MB=("kv_cache_MB", "mean"),
            net_latency_ms=("net_latency_ms", "mean"),
            kv_pct_of_ttft=("kv_pct_of_ttft", "mean"),
        ).round(2)
        if len(isl_groups) > 1:
            print("\n  KV overhead scaling with ISL:")
            print(isl_groups.to_string())
    else:
        print("  No AstraSim network latency in results (topology=none or AstraSim unavailable)")

    without_net = df[df["net_latency_ms"] == 0]
    if len(without_net) > 0:
        print(f"\n  Baseline configs (no network sim): {len(without_net)}")
        print(f"  TTFT range: {without_net['ttft_ms'].min():.1f} – {without_net['ttft_ms'].max():.1f} ms")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n  Results saved to {args.csv}")

    print()


if __name__ == "__main__":
    main()
