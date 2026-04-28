#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
32-GPU Comparison Plot — Best 5 configurations for 6 deployment modes.

Searches the configuration space for 32 GPUs across six modes:
  1. Agg (Standard)        — standard aggregated serving, shared GPUs
  2. Agg + AFD             — aggregated with attention-FFN disaggregation (M=1)
  3. Agg + AFD M=4         — aggregated AFD with 4-stage microbatch pipeline (M=4)
  4. Disagg (Standard)     — disaggregated prefill/decode, shared GPUs
  5. Disagg + AFD          — disaggregated prefill/decode with AFD (M=1)
  6. Disagg + AFD M=4      — disaggregated AFD with 4-stage microbatch pipeline (M=4)

For each mode, the top 5 configurations by seq/s are selected, and two
scatter plots are generated:
  - Top-5 by seq/s   (x = tpot, y = seq/s)
  - Top-5 by tokens/s (x = tpot, y = tokens/s)

Usage:
    cd /scratch1/hanjiang/aiconfigurator
    source aiconfigvenv/bin/activate
    python tools/plot_32gpu_comparison.py              # run search + plot
    python tools/plot_32gpu_comparison.py --from-csv   # reuse saved CSVs (skip search)
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Silence noisy warnings during the sweep ----------------------------
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ---- Project imports -----------------------------------------------------
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.pareto_analysis import (
    agg_pareto,
    agg_afd_pareto,
    disagg_pareto,
    disagg_afd_pareto,
)
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.utils import enumerate_parallel_config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# =========================================================================
# Configuration — same model/system as the replica and disagg studies
# =========================================================================
MODEL_PATH = "Qwen/Qwen3-30B-A3B"
SYSTEM = "h100_sxm"
BACKEND = "trtllm"
VERSION = "1.2.0rc5"

ISL = 4000
OSL = 500
PREFIX = 0
TTFT = 600.0
TPOT_LIST = list(range(1, 20, 1)) + list(range(20, 300, 5))

TOTAL_GPUS = 32
TOP_K = 5

# Disagg-specific tuning (matching afd_disagg_study.py)
PREFILL_LATENCY_CORRECTION = 1.1
DECODE_LATENCY_CORRECTION = 1.08
PREFILL_MAX_BS = 1
DECODE_MAX_BS = 512


# =========================================================================
# Helpers
# =========================================================================
def get_database() -> PerfDatabase:
    systems_dir = get_system_config_path()
    return PerfDatabase(SYSTEM, BACKEND, VERSION, systems_dir=str(systems_dir))


def get_base_model_config() -> config.ModelConfig:
    return config.ModelConfig(
        gemm_quant_mode=common.GEMMQuantMode.fp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8,
        comm_quant_mode=common.CommQuantMode.half,
    )


def get_runtime_config() -> config.RuntimeConfig:
    return config.RuntimeConfig(
        isl=ISL,
        osl=OSL,
        prefix=PREFIX,
        ttft=TTFT,
        tpot=TPOT_LIST,
    )


# =========================================================================
# Candidate worker sizes for replica-aware agg search
# =========================================================================
CANDIDATE_WORKER_SIZES = [1, 2, 4, 8, 16, 32]


def _divisors(budget: int, candidates: list[int]) -> list[int]:
    """Return candidates that evenly divide the budget."""
    return sorted(c for c in candidates if c <= budget and budget % c == 0)


# =========================================================================
# 1. Standard Agg search (replica-aware, matching afd_replica_study.py)
# =========================================================================
def search_agg_standard(database: PerfDatabase) -> pd.DataFrame:
    """Run standard agg_pareto at each valid worker size and scale by replicas.

    For 32 GPUs the database may not support very large ep values in a single
    worker.  Following the replica study approach we try worker sizes that
    evenly divide the GPU budget and pick the best per-worker config scaled
    by the number of replicas.
    """
    print(f"\n[1/6] Searching: Agg Standard (replica-aware) — {TOTAL_GPUS} GPUs ...")
    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    worker_sizes = _divisors(TOTAL_GPUS, CANDIDATE_WORKER_SIZES)
    print(f"  Worker sizes to try: {worker_sizes}")

    all_results = []
    for ws in worker_sizes:
        num_replicas = TOTAL_GPUS // ws
        max_dim = min(ws, 32)
        dim_list = sorted({2**i for i in range(6) if 2**i <= max_dim} | {ws})

        parallel_configs = enumerate_parallel_config(
            num_gpu_list=[ws],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=dim_list,
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=dim_list,
            is_moe=True,
            backend=common.BackendName.trtllm,
        )
        if not parallel_configs:
            continue

        try:
            df = agg_pareto(
                model_path=MODEL_PATH,
                runtime_config=runtime_config,
                database=copy.deepcopy(database),
                backend_name=BACKEND,
                model_config=model_config,
                parallel_config_list=parallel_configs,
            )
            if not df.empty:
                # Scale throughput metrics by number of replicas
                df = df.copy()
                df["num_replicas"] = num_replicas
                df["worker_gpus"] = ws
                for col in ["seq/s", "tokens/s"]:
                    if col in df.columns:
                        df[col] = df[col] * num_replicas
                # Recalculate per-GPU metrics based on total budget
                if "tokens/s" in df.columns:
                    df["tokens/s/gpu"] = df["tokens/s"] / TOTAL_GPUS
                if "seq/s" in df.columns:
                    df["seq/s/gpu"] = df["seq/s"] / TOTAL_GPUS
                # Ensure num_total_gpus reflects the full budget
                df["num_total_gpus"] = TOTAL_GPUS
                all_results.append(df)
                print(f"    {ws} GPU/worker × {num_replicas}R: {len(df)} rows")
        except Exception:
            print(f"    {ws} GPU/worker × {num_replicas}R: failed")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined.sort_values("tokens/s", ascending=False).reset_index(drop=True)
    print("  No valid configs found across any worker size.")
    return pd.DataFrame()


# =========================================================================
# 2. AFD Agg search (replica-aware)
# =========================================================================
def search_agg_afd(database: PerfDatabase) -> pd.DataFrame:
    """Run AFD agg_pareto at each valid worker size and scale by replicas."""
    print(f"\n[2/6] Searching: Agg + AFD (replica-aware) — {TOTAL_GPUS} GPUs ...")
    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    worker_sizes = _divisors(TOTAL_GPUS, CANDIDATE_WORKER_SIZES)
    print(f"  Worker sizes to try: {worker_sizes}")

    all_results = []
    for ws in worker_sizes:
        num_replicas = TOTAL_GPUS // ws
        max_dim = min(ws, 32)
        dim_list = sorted({2**i for i in range(6) if 2**i <= max_dim} | {ws})

        # Also add complement values for asymmetric AFD splits
        extra = set()
        for v in dim_list:
            c = ws - v
            if c > 0:
                extra.add(c)
        dim_list = sorted(set(dim_list) | extra)

        parallel_configs = enumerate_parallel_config(
            num_gpu_list=[ws],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=dim_list,
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=dim_list,
            is_moe=True,
            backend=common.BackendName.trtllm,
            enable_afd=True,
        )
        if not parallel_configs:
            continue

        try:
            df = agg_afd_pareto(
                model_path=MODEL_PATH,
                runtime_config=runtime_config,
                database=copy.deepcopy(database),
                backend_name=BACKEND,
                model_config=model_config,
                parallel_config_list=parallel_configs,
            )
            if not df.empty:
                df = df.copy()
                df["num_replicas"] = num_replicas
                df["worker_gpus"] = ws
                for col in ["seq/s", "tokens/s"]:
                    if col in df.columns:
                        df[col] = df[col] * num_replicas
                if "tokens/s" in df.columns:
                    df["tokens/s/gpu"] = df["tokens/s"] / TOTAL_GPUS
                if "seq/s" in df.columns:
                    df["seq/s/gpu"] = df["seq/s"] / TOTAL_GPUS
                df["num_total_gpus"] = TOTAL_GPUS
                all_results.append(df)
                print(f"    {ws} GPU/worker × {num_replicas}R: {len(df)} rows")
        except Exception:
            print(f"    {ws} GPU/worker × {num_replicas}R: failed")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined.sort_values("tokens/s", ascending=False).reset_index(drop=True)
    print("  No valid configs found across any worker size.")
    return pd.DataFrame()


# =========================================================================
# 3. Standard Disagg search
# =========================================================================
def search_disagg_standard(database: PerfDatabase) -> pd.DataFrame:
    """Run standard disagg_pareto for 32 GPUs."""
    print(f"\n[4/6] Searching: Disagg Standard — {TOTAL_GPUS} GPUs ...")
    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    tp_list = [1, 2]
    dp_list = [1, 2, 4, 8]
    moe_tp_list = [1]
    moe_ep_list = [1, 2, 4, 8]
    worker_gpu_list = [1, 2, 4, 8]

    prefill_configs = enumerate_parallel_config(
        num_gpu_list=worker_gpu_list,
        tp_list=tp_list,
        pp_list=[1],
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
    )
    decode_configs = enumerate_parallel_config(
        num_gpu_list=worker_gpu_list,
        tp_list=tp_list,
        pp_list=[1],
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
    )
    if not prefill_configs or not decode_configs:
        print("  No valid configs.")
        return pd.DataFrame()

    print(f"  Prefill: {len(prefill_configs)}, Decode: {len(decode_configs)} configs")
    replica_gpu_list = sorted({g for g in range(2, TOTAL_GPUS + 1)})

    try:
        return disagg_pareto(
            model_path=MODEL_PATH,
            runtime_config=runtime_config,
            prefill_database=copy.deepcopy(database),
            prefill_backend_name=BACKEND,
            prefill_model_config=model_config,
            prefill_parallel_config_list=prefill_configs,
            prefill_latency_correction_scale=PREFILL_LATENCY_CORRECTION,
            decode_database=copy.deepcopy(database),
            decode_backend_name=BACKEND,
            decode_model_config=model_config,
            decode_parallel_config_list=decode_configs,
            decode_latency_correction_scale=DECODE_LATENCY_CORRECTION,
            prefill_max_num_tokens=PREFILL_MAX_BS * ISL,
            decode_max_num_tokens=DECODE_MAX_BS,
            num_gpu_list=replica_gpu_list,
            max_num_gpu=TOTAL_GPUS,
            prefill_max_num_worker=32,
            decode_max_num_worker=32,
        )
    except Exception as e:
        print(f"  Failed: {e}")
        return pd.DataFrame()


# =========================================================================
# 4. AFD Disagg search
# =========================================================================
def search_disagg_afd(database: PerfDatabase) -> pd.DataFrame:
    """Run AFD disagg_pareto for 32 GPUs."""
    print(f"\n[5/6] Searching: Disagg + AFD — {TOTAL_GPUS} GPUs ...")
    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    tp_list = [1, 2]
    dp_list = [1, 2, 4, 8]
    moe_tp_list = [1]
    moe_ep_list = [1, 2, 4, 8]
    afd_worker_sizes = sorted({a + f
                               for t in tp_list for d in dp_list
                               for mt in moe_tp_list for me in moe_ep_list
                               for a in [t * d] for f in [mt * me]
                               if a + f <= 16})

    prefill_configs = enumerate_parallel_config(
        num_gpu_list=afd_worker_sizes,
        tp_list=tp_list,
        pp_list=[1],
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
        enable_afd=True,
    )
    decode_configs = enumerate_parallel_config(
        num_gpu_list=afd_worker_sizes,
        tp_list=tp_list,
        pp_list=[1],
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
        enable_afd=True,
    )
    if not prefill_configs or not decode_configs:
        print("  No valid configs.")
        return pd.DataFrame()

    print(f"  Prefill AFD: {len(prefill_configs)}, Decode AFD: {len(decode_configs)} configs")
    replica_gpu_list = sorted({g for g in range(2, TOTAL_GPUS + 1)})

    try:
        return disagg_afd_pareto(
            model_path=MODEL_PATH,
            runtime_config=runtime_config,
            prefill_database=copy.deepcopy(database),
            prefill_backend_name=BACKEND,
            prefill_model_config=model_config,
            prefill_parallel_config_list=prefill_configs,
            prefill_latency_correction_scale=PREFILL_LATENCY_CORRECTION,
            decode_database=copy.deepcopy(database),
            decode_backend_name=BACKEND,
            decode_model_config=model_config,
            decode_parallel_config_list=decode_configs,
            decode_latency_correction_scale=DECODE_LATENCY_CORRECTION,
            prefill_max_num_tokens=PREFILL_MAX_BS * ISL,
            decode_max_num_tokens=DECODE_MAX_BS,
            num_gpu_list=replica_gpu_list,
            max_num_gpu=TOTAL_GPUS,
            prefill_max_num_worker=32,
            decode_max_num_worker=32,
        )
    except Exception as e:
        print(f"  Failed: {e}")
        return pd.DataFrame()


# =========================================================================
# 3. AFD M=4 Agg search
# =========================================================================
def search_agg_afd_m4(database: PerfDatabase) -> pd.DataFrame:
    """Run AFD M=4 agg_pareto (4-stage pipeline) at each valid worker size and scale by replicas."""
    print(f"\n[3/6] Searching: Agg + AFD M=4 (replica-aware) — {TOTAL_GPUS} GPUs ...")
    model_config = get_base_model_config()
    model_config.afd_num_microbatches = 4
    runtime_config = get_runtime_config()

    worker_sizes = _divisors(TOTAL_GPUS, CANDIDATE_WORKER_SIZES)
    print(f"  Worker sizes to try: {worker_sizes}")

    all_results = []
    for ws in worker_sizes:
        num_replicas = TOTAL_GPUS // ws
        max_dim = min(ws, 32)
        dim_list = sorted({2**i for i in range(6) if 2**i <= max_dim} | {ws})

        extra = set()
        for v in dim_list:
            c = ws - v
            if c > 0:
                extra.add(c)
        dim_list = sorted(set(dim_list) | extra)

        parallel_configs = enumerate_parallel_config(
            num_gpu_list=[ws],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=dim_list,
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=dim_list,
            is_moe=True,
            backend=common.BackendName.trtllm,
            enable_afd=True,
        )
        if not parallel_configs:
            continue

        try:
            df = agg_afd_pareto(
                model_path=MODEL_PATH,
                runtime_config=runtime_config,
                database=copy.deepcopy(database),
                backend_name=BACKEND,
                model_config=model_config,
                parallel_config_list=parallel_configs,
            )
            if not df.empty:
                df = df.copy()
                df["num_replicas"] = num_replicas
                df["worker_gpus"] = ws
                for col in ["seq/s", "tokens/s"]:
                    if col in df.columns:
                        df[col] = df[col] * num_replicas
                if "tokens/s" in df.columns:
                    df["tokens/s/gpu"] = df["tokens/s"] / TOTAL_GPUS
                if "seq/s" in df.columns:
                    df["seq/s/gpu"] = df["seq/s"] / TOTAL_GPUS
                df["num_total_gpus"] = TOTAL_GPUS
                all_results.append(df)
                print(f"    {ws} GPU/worker × {num_replicas}R: {len(df)} rows")
        except Exception:
            print(f"    {ws} GPU/worker × {num_replicas}R: failed")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        return combined.sort_values("tokens/s", ascending=False).reset_index(drop=True)
    print("  No valid configs found across any worker size.")
    return pd.DataFrame()


# =========================================================================
# 6. AFD M=4 Disagg search
# =========================================================================
def search_disagg_afd_m4(database: PerfDatabase) -> pd.DataFrame:
    """Run AFD M=4 disagg_pareto (4-stage pipeline) for the full GPU budget."""
    print(f"\n[6/6] Searching: Disagg + AFD M=4 — {TOTAL_GPUS} GPUs ...")
    model_config = get_base_model_config()
    model_config.afd_num_microbatches = 4
    runtime_config = get_runtime_config()

    tp_list = [1, 2]
    dp_list = [1, 2, 4, 8]
    moe_tp_list = [1]
    moe_ep_list = [1, 2, 4, 8]
    afd_worker_sizes = sorted({a + f
                               for t in tp_list for d in dp_list
                               for mt in moe_tp_list for me in moe_ep_list
                               for a in [t * d] for f in [mt * me]
                               if a + f <= 16})

    prefill_configs = enumerate_parallel_config(
        num_gpu_list=afd_worker_sizes,
        tp_list=tp_list,
        pp_list=[1],
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
        enable_afd=True,
    )
    decode_configs = enumerate_parallel_config(
        num_gpu_list=afd_worker_sizes,
        tp_list=tp_list,
        pp_list=[1],
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
        enable_afd=True,
    )
    if not prefill_configs or not decode_configs:
        print("  No valid configs.")
        return pd.DataFrame()

    print(f"  Prefill AFD M=4: {len(prefill_configs)}, Decode AFD M=4: {len(decode_configs)} configs")
    replica_gpu_list = sorted({g for g in range(2, TOTAL_GPUS + 1)})

    try:
        return disagg_afd_pareto(
            model_path=MODEL_PATH,
            runtime_config=runtime_config,
            prefill_database=copy.deepcopy(database),
            prefill_backend_name=BACKEND,
            prefill_model_config=model_config,
            prefill_parallel_config_list=prefill_configs,
            prefill_latency_correction_scale=PREFILL_LATENCY_CORRECTION,
            decode_database=copy.deepcopy(database),
            decode_backend_name=BACKEND,
            decode_model_config=model_config,
            decode_parallel_config_list=decode_configs,
            decode_latency_correction_scale=DECODE_LATENCY_CORRECTION,
            prefill_max_num_tokens=PREFILL_MAX_BS * ISL,
            decode_max_num_tokens=DECODE_MAX_BS,
            num_gpu_list=replica_gpu_list,
            max_num_gpu=TOTAL_GPUS,
            prefill_max_num_worker=32,
            decode_max_num_worker=32,
        )
    except Exception as e:
        print(f"  Failed: {e}")
        return pd.DataFrame()


# =========================================================================
# Extract top-K rows for a given metric
# =========================================================================
def extract_top_k(
    df: pd.DataFrame, metric: str, k: int = TOP_K, ascending: bool = False,
) -> pd.DataFrame:
    """Return the top-k rows sorted by `metric`.

    Args:
        ascending: If True, return the *lowest* k values (useful for latency).
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    sorted_df = (
        df.sort_values(by=metric, ascending=ascending)
        .head(k)
    )
    return sorted_df.reset_index(drop=True)


# =========================================================================
# Plotting
# =========================================================================
# Visual config for each mode: (color, marker, label)
MODE_STYLES = {
    "Agg Standard":      {"color": "#2196F3", "marker": "o",  "label": "Agg (Standard)"},
    "Agg AFD":           {"color": "#FF9800", "marker": "s",  "label": "Agg + AFD (M=1)"},
    "Agg AFD M=4":       {"color": "#FF5722", "marker": "P",  "label": "Agg + AFD M=4"},
    "Disagg Standard":   {"color": "#4CAF50", "marker": "^",  "label": "Disagg (Standard)"},
    "Disagg AFD":        {"color": "#E91E63", "marker": "D",  "label": "Disagg + AFD (M=1)"},
    "Disagg AFD M=4":    {"color": "#9C27B0", "marker": "*",  "label": "Disagg + AFD M=4"},
}


def _build_scatter_data(
    mode_dfs: dict[str, pd.DataFrame],
    y_metric: str,
    x_metric: str = "tpot",
    ascending: bool = False,
) -> dict[str, dict]:
    """Build per-mode scatter data dicts with x, y, and annotations."""
    scatter = {}
    for mode_key, df in mode_dfs.items():
        top = extract_top_k(df, y_metric, TOP_K, ascending=ascending)
        if top.empty:
            continue
        x = top[x_metric].values if x_metric in top.columns else np.arange(len(top))
        y = top[y_metric].values

        # Build annotation labels depending on mode type
        annotations = []
        for _, row in top.iterrows():
            if "(p)tp" in top.columns:
                # Disagg mode
                ann = (f"P:tp{int(row.get('(p)tp',0))}dp{int(row.get('(p)dp',0))}"
                       f"ep{int(row.get('(p)moe_ep',0))}×{int(row.get('(p)workers',0))}w\n"
                       f"D:tp{int(row.get('(d)tp',0))}dp{int(row.get('(d)dp',0))}"
                       f"ep{int(row.get('(d)moe_ep',0))}×{int(row.get('(d)workers',0))}w")
            else:
                # Agg mode
                ann = (f"tp{int(row.get('tp',0))} dp{int(row.get('dp',0))} "
                       f"ep{int(row.get('moe_ep',0))} bs{int(row.get('bs',0))}")
            annotations.append(ann)

        scatter[mode_key] = {"x": x, "y": y, "annotations": annotations}
    return scatter


def plot_metric(
    mode_dfs: dict[str, pd.DataFrame],
    y_metric: str,
    y_label: str,
    title: str,
    filename: str,
    x_metric: str = "tpot",
    x_label: str = "TPOT (ms)",
    ascending: bool = False,
):
    """Create a scatter plot for 6 modes showing top-K points."""
    scatter = _build_scatter_data(mode_dfs, y_metric, x_metric, ascending=ascending)
    if not scatter:
        print(f"  No data to plot for {y_metric}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    for mode_key, data in scatter.items():
        style = MODE_STYLES[mode_key]
        ax.scatter(
            data["x"], data["y"],
            c=style["color"],
            marker=style["marker"],
            s=160,
            edgecolors="black",
            linewidths=0.5,
            label=style["label"],
            zorder=5,
        )
        # Annotate each point
        for i, ann in enumerate(data["annotations"]):
            ax.annotate(
                ann,
                (data["x"][i], data["y"][i]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=6,
                alpha=0.85,
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


def print_top_k_table(
    mode_dfs: dict[str, pd.DataFrame],
    metric: str,
    label: str,
    ascending: bool = False,
):
    """Print a summary table for the top-K of each mode."""
    print(f"\n{'='*100}")
    print(f"  TOP {TOP_K} by {label}  ({TOTAL_GPUS} GPUs, {MODEL_PATH})")
    print(f"{'='*100}")

    for mode_key, df in mode_dfs.items():
        top = extract_top_k(df, metric, TOP_K, ascending=ascending)
        if top.empty:
            print(f"\n  [{mode_key}] — no results")
            continue

        print(f"\n  [{mode_key}]")
        # Agg columns vs disagg columns
        if "(p)tp" in top.columns:
            cols = [
                "(p)tp", "(p)dp", "(p)moe_ep", "(p)workers",
                "(d)tp", "(d)dp", "(d)moe_ep", "(d)workers",
                "num_total_gpus", "seq/s", "tokens/s", "tokens/s/gpu",
                "tpot", "ttft", "request_latency",
            ]
        else:
            cols = [
                "tp", "dp", "moe_tp", "moe_ep",
                "num_total_gpus", "num_attn_gpus", "num_ffn_gpus",
                "bs", "seq/s", "tokens/s", "tokens/s/gpu",
                "tpot", "ttft", "request_latency", "memory",
            ]
        avail = [c for c in cols if c in top.columns]
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(top[avail].to_string(index=True))


# =========================================================================
# CSV I/O
# =========================================================================
CSV_PREFIX = "plot_32gpu"


def save_csvs(mode_dfs: dict[str, pd.DataFrame]):
    """Save each mode's full result DataFrame to CSV."""
    for mode_key, df in mode_dfs.items():
        if df.empty:
            continue
        safe_name = mode_key.replace(" ", "_").replace("=", "").lower()
        fname = f"{CSV_PREFIX}_{safe_name}_{SYSTEM}.csv"
        df.to_csv(fname, index=False)
        print(f"  Saved CSV: {fname} ({len(df)} rows)")


def load_csvs() -> dict[str, pd.DataFrame]:
    """Load previously saved CSVs."""
    mode_dfs = {}
    for mode_key in MODE_STYLES:
        safe_name = mode_key.replace(" ", "_").replace("=", "").lower()
        fname = f"{CSV_PREFIX}_{safe_name}_{SYSTEM}.csv"
        if os.path.exists(fname):
            mode_dfs[mode_key] = pd.read_csv(fname)
            print(f"  Loaded: {fname} ({len(mode_dfs[mode_key])} rows)")
        else:
            print(f"  Not found: {fname} — skipping {mode_key}")
            mode_dfs[mode_key] = pd.DataFrame()
    return mode_dfs


# =========================================================================
# Main
# =========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top-5 configs for 6 deployment modes on 32 GPUs (incl. AFD M=4).",
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Skip the search and load results from previously saved CSVs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"32-GPU Comparison: {MODEL_PATH} on {SYSTEM}")
    print(f"  ISL={ISL}, OSL={OSL}, TTFT={TTFT}ms, Backend={BACKEND} {VERSION}")
    print(f"  Top-K = {TOP_K}")

    if args.from_csv:
        print("\n  Loading from saved CSVs ...")
        mode_dfs = load_csvs()
    else:
        database = get_database()
        mode_dfs = {
            "Agg Standard":    search_agg_standard(database),
            "Agg AFD":         search_agg_afd(database),
            "Agg AFD M=4":     search_agg_afd_m4(database),
            "Disagg Standard": search_disagg_standard(database),
            "Disagg AFD":      search_disagg_afd(database),
            "Disagg AFD M=4":  search_disagg_afd_m4(database),
        }
        save_csvs(mode_dfs)

    # ---- Print summary tables ----
    print_top_k_table(mode_dfs, "seq/s", "seq/s")
    print_top_k_table(mode_dfs, "tokens/s", "tokens/s")
    print_top_k_table(mode_dfs, "request_latency", "request_latency (ms, lower=better)",
                      ascending=True)
    print_top_k_table(mode_dfs, "ttft", "TTFT (ms, lower=better)", ascending=True)

    # ---- Plot: Top-5 by seq/s ----
    plot_metric(
        mode_dfs,
        y_metric="seq/s",
        y_label="seq/s",
        title=f"Top-{TOP_K} Configs by seq/s — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
        filename=f"plot_32gpu_top5_seqs_{SYSTEM}.png",
    )

    # ---- Plot: Top-5 by tokens/s ----
    plot_metric(
        mode_dfs,
        y_metric="tokens/s",
        y_label="tokens/s",
        title=f"Top-{TOP_K} Configs by tokens/s — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
        filename=f"plot_32gpu_top5_tokenss_{SYSTEM}.png",
    )

    # ---- Plot: Top-5 by E2E request latency (lower is better) ----
    plot_metric(
        mode_dfs,
        y_metric="request_latency",
        y_label="E2E Request Latency (ms)",
        title=f"Top-{TOP_K} Fastest Configs by E2E Latency — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
        filename=f"plot_32gpu_top5_e2e_latency_{SYSTEM}.png",
        ascending=True,
    )

    # ---- Plot: Top-5 by TTFT (lower is better) ----
    plot_metric(
        mode_dfs,
        y_metric="ttft",
        y_label="TTFT (ms)",
        title=f"Top-{TOP_K} Fastest Configs by TTFT — {TOTAL_GPUS} GPUs ({MODEL_PATH})",
        filename=f"plot_32gpu_top5_ttft_{SYSTEM}.png",
        ascending=True,
    )

    print(f"\nDone. Plots saved as:")
    print(f"  plot_32gpu_top5_seqs_{SYSTEM}.png")
    print(f"  plot_32gpu_top5_tokenss_{SYSTEM}.png")
    print(f"  plot_32gpu_top5_e2e_latency_{SYSTEM}.png")
    print(f"  plot_32gpu_top5_ttft_{SYSTEM}.png")
    print(f"  (6 modes: Agg Std, Agg AFD M=1, Agg AFD M=4, Disagg Std, Disagg AFD M=1, Disagg AFD M=4)")


if __name__ == "__main__":
    main()
