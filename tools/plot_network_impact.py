#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Network Impact Plot — Real TTFT + KV Transfer from DisaggInferenceSession.run_disagg().

Runs actual disaggregated inference configurations through the SDK and plots:
  1. Stacked bar chart: Compute TTFT + KV transfer latency per parallelism config
  2. Grouped bar chart: KV transfer latency across ISL values (cross-node vs same-node)

All TTFT and KV transfer numbers come from the real PerfDatabase and AstraSim
simulation — **no hardcoded or roofline estimates**.

Usage:
    cd /scratch1/hanjiang/aiconfigurator
    source aiconfigvenv/bin/activate
    python tools/plot_network_impact.py
"""
from __future__ import annotations

import copy
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.astrasim_utils import AstraSimManager, NETWORK_SIM_AVAILABLE
from aiconfigurator.sdk.inference_session import DisaggInferenceSession
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.backends.factory import get_backend

# =========================================================================
# Configuration — matches plot_8gpu_comparison.py
# =========================================================================
MODEL_PATH = "Qwen/Qwen3-30B-A3B"
SYSTEM = "h100_sxm"
BACKEND = "trtllm"
VERSION = "1.2.0rc5"

ISL = 4000
OSL = 500
PREFIX = 0
TTFT_CONSTRAINT = 600.0
TPOT_LIST = list(range(1, 20, 1)) + list(range(20, 300, 5))

PREFILL_LATENCY_CORRECTION = 1.1
DECODE_LATENCY_CORRECTION = 1.08


# =========================================================================
# Helpers
# =========================================================================
def get_database() -> PerfDatabase:
    return PerfDatabase(SYSTEM, BACKEND, VERSION, systems_dir=str(get_system_config_path()))


def get_base_model_config(**overrides) -> config.ModelConfig:
    mc = config.ModelConfig(
        gemm_quant_mode=common.GEMMQuantMode.fp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8,
        comm_quant_mode=common.CommQuantMode.half,
    )
    for k, v in overrides.items():
        setattr(mc, k, v)
    return mc


def get_runtime_config(isl: int = ISL) -> config.RuntimeConfig:
    return config.RuntimeConfig(
        isl=isl,
        osl=OSL,
        prefix=PREFIX,
        ttft=TTFT_CONSTRAINT,
        tpot=TPOT_LIST,
    )


def run_one_disagg(
    database: PerfDatabase,
    p_tp: int, p_moe_tp: int, p_moe_ep: int, p_bs: int, p_workers: int,
    d_tp: int, d_moe_tp: int, d_moe_ep: int, d_bs: int, d_workers: int,
    isl: int = ISL,
) -> dict | None:
    """Run a single disagg config and return TTFT breakdown.

    Returns dict with: compute_ttft, kv_network_latency_ms, total_ttft, tpot,
    request_latency, kv_cache_size_bytes, num_total_gpus, label, etc.
    Returns None on failure.
    """
    db = copy.deepcopy(database)
    prefill_backend = get_backend(BACKEND)
    decode_backend = get_backend(BACKEND)

    prefill_mc = get_base_model_config(tp_size=p_tp, moe_tp_size=p_moe_tp, moe_ep_size=p_moe_ep)
    decode_mc = get_base_model_config(tp_size=d_tp, moe_tp_size=d_moe_tp, moe_ep_size=d_moe_ep)
    runtime_cfg = get_runtime_config(isl=isl)

    sess = DisaggInferenceSession(
        prefill_database=db,
        prefill_backend=prefill_backend,
        decode_database=copy.deepcopy(db),
        decode_backend=decode_backend,
    )
    sess.set_latency_correction_scales(PREFILL_LATENCY_CORRECTION, DECODE_LATENCY_CORRECTION)

    try:
        summary = sess.run_disagg(
            model_path=MODEL_PATH,
            runtime_config=runtime_cfg,
            prefill_model_config=prefill_mc,
            prefill_batch_size=p_bs,
            prefill_num_worker=p_workers,
            decode_model_config=decode_mc,
            decode_batch_size=d_bs,
            decode_num_worker=d_workers,
        )
    except Exception as e:
        return None

    df = summary.get_summary_df()
    if df.empty:
        return None

    row = df.iloc[0]
    total_ttft = float(row["ttft"])
    tpot = float(row["tpot"])
    request_latency = float(row["request_latency"])

    kv_lat = float(row.get("kv_network_latency_ms", 0.0))
    kv_bytes = float(row.get("kv_cache_size_bytes", 0.0))
    compute_ttft = total_ttft - kv_lat  # compute-only TTFT

    total_gpus = int(row["num_total_gpus"])

    return {
        "compute_ttft": compute_ttft,
        "kv_network_latency_ms": kv_lat,
        "total_ttft": total_ttft,
        "tpot": tpot,
        "request_latency": request_latency,
        "kv_cache_size_bytes": kv_bytes,
        "num_total_gpus": total_gpus,
        "p_tp": p_tp, "p_moe_tp": p_moe_tp, "p_moe_ep": p_moe_ep,
        "p_bs": p_bs, "p_workers": p_workers,
        "d_tp": d_tp, "d_moe_tp": d_moe_tp, "d_moe_ep": d_moe_ep,
        "d_bs": d_bs, "d_workers": d_workers,
        "isl": isl,
    }


# =========================================================================
# Main
# =========================================================================
def main():
    if not NETWORK_SIM_AVAILABLE:
        print("ERROR: AstraSim C++ library not available.")
        sys.exit(1)

    print(f"Network Impact Plot: {MODEL_PATH} on {SYSTEM}")
    print(f"  ISL={ISL}, OSL={OSL}, Backend={BACKEND} {VERSION}")
    print()

    database = get_database()

    # ── Define disagg configs to sweep ──
    # Each tuple: (label, p_tp, p_moe_tp, p_moe_ep, p_bs, p_workers,
    #                     d_tp, d_moe_tp, d_moe_ep, d_bs, d_workers)
    #
    # H100 SXM: 8 GPUs/node.  Contiguous GPU layout means:
    #   - Total prefill GPUs = p_workers × p_tp → fills first N GPUs
    #   - Decode starts right after → may land on next node if >8 GPUs used
    # So configs with total_prefill_GPUs ≥ 8 push decode to node 1 → IB transfers.
    CONFIGS = [
        # ── Same-node only (all GPUs on node 0, NVLink) ──
        ("1P+7D TP1\n(8 GPU, same-node)",   1, 1, 1, 1, 1,  1, 1, 1, 2, 7),
        ("1P+3D TP2\n(8 GPU, same-node)",   2, 1, 2, 1, 1,  2, 1, 2, 4, 3),
        ("2P+2D TP2\n(8 GPU, same-node)",   2, 1, 2, 1, 2,  2, 1, 2, 4, 2),
        ("2P+6D TP2\n(16 GPU, same-node)",  2, 1, 2, 1, 2,  2, 1, 2, 4, 6),  # P on GPUs 0-3 → D starts 4 (still node 0)
        # ── Cross-node (decode lands on node 1, IB) ──
        ("4P+4D TP2\n(16 GPU, cross-node)", 2, 1, 2, 1, 4,  2, 1, 2, 4, 4),  # P fills GPUs 0-7 → D starts at GPU 8
        ("8P+8D TP1\n(16 GPU, cross-node)", 1, 1, 1, 1, 8,  1, 1, 1, 2, 8),  # P fills GPUs 0-7 → D at 8-15
        ("8P+8D TP2\n(32 GPU, cross-node)", 2, 1, 2, 1, 8,  2, 1, 2, 8, 8),  # P fills GPUs 0-15 → D at 16-31
    ]

    # ── Run all configs ──
    results = []
    for i, (label, *args) in enumerate(CONFIGS):
        print(f"  [{i+1}/{len(CONFIGS)}] Running {label.replace(chr(10), ' ')} ...", end=" ", flush=True)
        r = run_one_disagg(database, *args)
        if r is not None:
            r["label"] = label
            results.append(r)
            print(f"TTFT={r['total_ttft']:.2f}ms (compute={r['compute_ttft']:.2f} + KV={r['kv_network_latency_ms']:.2f})")
        else:
            print("FAILED")

    if not results:
        print("No configs succeeded. Exiting.")
        sys.exit(1)

    # ── Colors ──
    BLUE = "#2563EB"
    ORANGE = "#F59E0B"
    RED = "#EF4444"
    GREEN = "#10B981"
    LIGHT_BLUE = "#93C5FD"

    # ================================================================
    # FIGURE 1: Stacked bar — Compute TTFT + KV Transfer per config
    # ================================================================
    labels = [r["label"] for r in results]
    compute_ttfts = [r["compute_ttft"] for r in results]
    kv_lats = [r["kv_network_latency_ms"] for r in results]
    total_ttfts = [r["total_ttft"] for r in results]

    fig1, ax1 = plt.subplots(figsize=(max(10, len(results) * 1.6), 6))
    x = np.arange(len(results))
    width = 0.55

    ax1.bar(x, compute_ttfts, width, label="Compute TTFT", color=BLUE,
            edgecolor="white", linewidth=0.5)
    ax1.bar(x, kv_lats, width, bottom=compute_ttfts,
            label="KV Transfer (AstraSim)", color=ORANGE,
            edgecolor="white", linewidth=0.5)

    max_total = max(total_ttfts) if total_ttfts else 1
    for i, r in enumerate(results):
        c, n = r["compute_ttft"], r["kv_network_latency_ms"]
        # Compute label inside blue bar
        if c > max_total * 0.06:
            ax1.text(i, c / 2, f"{c:.1f}", ha="center", va="center", fontsize=8.5,
                     fontweight="bold", color="white")
        # KV label
        if n > max_total * 0.04:
            ax1.text(i, c + n / 2, f"{n:.1f}", ha="center", va="center", fontsize=8.5,
                     fontweight="bold", color="white")
        elif n > 0.01:
            ax1.text(i, c + n + max_total * 0.02, f"{n:.2f}", ha="center", va="bottom",
                     fontsize=7, fontweight="bold", color=ORANGE)
        # Total on top
        ax1.text(i, c + n + max_total * 0.05,
                 f"{c + n:.1f} ms", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#1F2937")

    ax1.set_ylabel("Time to First Token (ms)", fontsize=12)
    ax1.set_title(
        f"TTFT Breakdown: Compute + KV Transfer across Parallelism Configs\n"
        f"{MODEL_PATH}, ISL={ISL}, OSL={OSL} — {SYSTEM}  (from run_disagg)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, max_total * 1.25)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    fig1.tight_layout()
    out1 = "plot_network_impact_ttft.png"
    fig1.savefig(out1, dpi=180, bbox_inches="tight")
    print(f"\n[1/2] Saved: {os.path.abspath(out1)}")
    plt.close(fig1)

    # ================================================================
    # FIGURE 2: KV Transfer Latency across ISL values
    # Pick two representative configs: one same-node, one cross-node
    # ================================================================
    # Use the same-node config: 1P+7D TP1 (all 8 GPUs same node)
    # and cross-node config: 1P+3D TP2 (if decode spans cross-node)
    # We'll run a sweep of ISL values
    ISL_VALUES = [500, 1000, 2000, 4000, 8000]

    # Pick two configs: same-node (8 GPU) vs cross-node (16 GPU)
    cfg_same = ("1P+7D TP1 (same-node)",  1, 1, 1, 1, 1,  1, 1, 1, 2, 7)  # all NVLink
    cfg_cross = ("4P+4D TP2 (cross-node)", 2, 1, 2, 1, 4,  2, 1, 2, 4, 4)  # spans node boundary

    same_kv = []
    cross_kv = []
    same_ttft_total = []
    cross_ttft_total = []

    print(f"\n  Sweeping ISL values: {ISL_VALUES}")
    for isl in ISL_VALUES:
        print(f"    ISL={isl} ...", end=" ", flush=True)
        r1 = run_one_disagg(database, *cfg_same[1:], isl=isl)
        r2 = run_one_disagg(database, *cfg_cross[1:], isl=isl)

        kv1 = r1["kv_network_latency_ms"] if r1 else 0
        kv2 = r2["kv_network_latency_ms"] if r2 else 0
        same_kv.append(kv1)
        cross_kv.append(kv2)
        same_ttft_total.append(r1["total_ttft"] if r1 else 0)
        cross_ttft_total.append(r2["total_ttft"] if r2 else 0)
        print(f"{cfg_same[0].split(chr(10))[0]} KV={kv1:.2f}ms, "
              f"{cfg_cross[0].split(chr(10))[0]} KV={kv2:.2f}ms")

    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    x2 = np.arange(len(ISL_VALUES))
    w = 0.35

    ax2.bar(x2 - w / 2, same_kv, w, label=cfg_same[0].replace("\n", " "),
            color=GREEN, edgecolor="white", linewidth=0.5)
    ax2.bar(x2 + w / 2, cross_kv, w, label=cfg_cross[0].replace("\n", " "),
            color=RED, edgecolor="white", linewidth=0.5)

    for i, (s, c) in enumerate(zip(same_kv, cross_kv)):
        if s > 0.001:
            ax2.text(i - w / 2, s + 0.05, f"{s:.2f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=GREEN)
        if c > 0.001:
            ax2.text(i + w / 2, c + 0.05, f"{c:.2f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=RED)

    ax2.set_xlabel("Input Sequence Length (ISL)", fontsize=12)
    ax2.set_ylabel("KV Transfer Latency (ms)", fontsize=12)
    ax2.set_title(
        f"KV Cache Transfer Latency across ISL Values\n"
        f"{MODEL_PATH} — {SYSTEM}  (from run_disagg)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(v) for v in ISL_VALUES], fontsize=11)
    all_kv = same_kv + cross_kv
    max_kv = max(all_kv) if all_kv else 1
    ax2.set_ylim(0, max_kv * 1.3 if max_kv > 0.01 else 1.0)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    fig2.tight_layout()
    out2 = "plot_network_impact_isl.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"\n[2/2] Saved: {os.path.abspath(out2)}")
    plt.close(fig2)

    # ── Print summary table ──
    print(f"\n{'='*100}")
    print(f"  Summary: TTFT Breakdown for {MODEL_PATH} on {SYSTEM}")
    print(f"{'='*100}")
    print(f"  {'Config':<24s} {'GPUs':>5s} {'Compute TTFT':>13s} {'KV Transfer':>12s} "
          f"{'Total TTFT':>11s} {'TPOT':>7s} {'E2E Latency':>12s}")
    print(f"  {'':24s} {'':>5s} {'(ms)':>13s} {'(ms)':>12s} "
          f"{'(ms)':>11s} {'(ms)':>7s} {'(ms)':>12s}")
    print("  " + "─" * 92)
    for r in results:
        lbl = r["label"].replace("\n", " ")
        print(f"  {lbl:<24s} {r['num_total_gpus']:>5d} {r['compute_ttft']:>13.2f} "
              f"{r['kv_network_latency_ms']:>12.2f} {r['total_ttft']:>11.2f} "
              f"{r['tpot']:>7.3f} {r['request_latency']:>12.2f}")

    print(f"\nDone! Plots saved as {out1} and {out2}")


if __name__ == "__main__":
    main()
