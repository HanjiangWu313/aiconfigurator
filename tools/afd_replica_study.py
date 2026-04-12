#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replica AFD study for Qwen3-30B-A3B.

For each GPU budget it tries every *worker size* that evenly divides the
budget and computes:

    total_tokens_s = per_worker_tokens_s × num_replicas

Both standard (shared GPUs) and AFD (decoupled attn/FFN GPUs)
configurations are evaluated at each worker size.

Output:
    - Per-budget comparison tables printed to stdout
    - CSV files: replica_study_{budget}gpu.csv
"""

from __future__ import annotations

import copy
import logging
import math
import sys
import warnings

import pandas as pd

# ---- Silence noisy warnings during the sweep ----------------------------
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ---- Project imports -----------------------------------------------------
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.pareto_analysis import agg_pareto, agg_afd_pareto, get_pareto_front
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.utils import enumerate_parallel_config

# =========================================================================
# Configuration
# =========================================================================
MODEL_PATH = "Qwen/Qwen3-30B-A3B"
SYSTEM = "b200_sxm"
BACKEND = "trtllm"
VERSION = "1.2.0rc5"

ISL = 4000
OSL = 500
PREFIX = 0
TTFT = 600.0
TPOT_LIST = list(range(1, 20, 1)) + list(range(20, 300, 5))

GPU_BUDGETS = [8, 16, 32, 64]

# Worker sizes to consider.  The script will only use sizes that evenly
# divide the GPU budget.  Keep this list reasonable — the perf database
# may not have data for very large ep values.
CANDIDATE_WORKER_SIZES = [1, 2, 4, 8, 16, 32]


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


def _divisors(budget: int, candidates: list[int]) -> list[int]:
    """Return candidates that evenly divide the budget."""
    return sorted(c for c in candidates if c <= budget and budget % c == 0)


# =========================================================================
# Run a single worker-size experiment (standard or AFD)
# =========================================================================
def run_worker_pareto(
    database: PerfDatabase,
    worker_gpus: int,
    *,
    enable_afd: bool = False,
) -> pd.DataFrame:
    """Run agg_pareto for one worker size. Returns the full result DF."""

    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    max_dim = min(worker_gpus, 32)
    dim_list = sorted({2**i for i in range(6) if 2**i <= max_dim} | {worker_gpus})

    if enable_afd:
        # Also add complement values so we explore asymmetric splits
        extra = set()
        for v in dim_list:
            c = worker_gpus - v
            if c > 0:
                extra.add(c)
        dim_list = sorted(set(dim_list) | extra)

    parallel_configs = enumerate_parallel_config(
        num_gpu_list=[worker_gpus],
        tp_list=[1, 2, 4, 8],
        pp_list=[1],
        dp_list=dim_list,
        moe_tp_list=[1, 2, 4, 8],
        moe_ep_list=dim_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
        enable_afd=enable_afd,
    )

    if not parallel_configs:
        return pd.DataFrame()

    pareto_fn = agg_afd_pareto if enable_afd else agg_pareto
    try:
        return pareto_fn(
            model_path=MODEL_PATH,
            runtime_config=runtime_config,
            database=copy.deepcopy(database),
            backend_name=BACKEND,
            model_config=model_config,
            parallel_config_list=parallel_configs,
        )
    except Exception:
        return pd.DataFrame()


# =========================================================================
# Replica-aware analysis for one GPU budget
# =========================================================================
def analyse_budget(database: PerfDatabase, total_gpus: int) -> pd.DataFrame:
    """
    For each valid worker size, compute per-worker best config & scale by
    number of replicas.  Returns a combined DF with columns for total throughput.
    """
    worker_sizes = _divisors(total_gpus, CANDIDATE_WORKER_SIZES)

    rows: list[dict] = []

    for ws in worker_sizes:
        num_replicas = total_gpus // ws

        for mode, afd_flag in [("Standard", False), ("AFD", True)]:
            label = f"{mode} {ws}gpu × {num_replicas}R"
            print(f"    Running {label} ...", end="", flush=True)

            df = run_worker_pareto(database, ws, enable_afd=afd_flag)

            if df.empty:
                print(" no results")
                continue

            # Pick the best config by tokens/s (per-worker overall throughput)
            best = df.loc[df["tokens/s"].idxmax()]

            total_tok_s = best["tokens/s"] * num_replicas
            total_seq_s = best.get("seq/s", 0) * num_replicas

            row = {
                "mode": mode,
                "worker_gpus": ws,
                "num_replicas": num_replicas,
                "total_gpus": total_gpus,
                "tp": int(best["tp"]),
                "dp": int(best["dp"]),
                "moe_tp": int(best["moe_tp"]),
                "moe_ep": int(best["moe_ep"]),
                "bs_per_worker": int(best["bs"]),
                "global_bs_per_worker": int(best.get("global_bs", best["bs"])),
                "global_bs_total": int(best.get("global_bs", best["bs"])) * num_replicas,
                "tokens/s (worker)": round(best["tokens/s"], 1),
                "tokens/s (total)": round(total_tok_s, 1),
                "tokens/s/gpu": round(total_tok_s / total_gpus, 1),
                "tokens/s/user": round(best.get("tokens/s/user", 0), 3),
                "tpot": round(best.get("tpot", 0), 3),
                "ttft": round(best.get("ttft", 0), 3),
                "memory": round(best.get("memory", 0), 3),
            }

            # Add AFD-specific columns
            if afd_flag and "num_attn_gpus" in df.columns:
                row["attn_gpus/worker"] = int(best["num_attn_gpus"])
                row["ffn_gpus/worker"] = int(best["num_ffn_gpus"])
            else:
                row["attn_gpus/worker"] = ws
                row["ffn_gpus/worker"] = ws

            rows.append(row)
            print(f" {total_tok_s:,.0f} tok/s total ({best['tokens/s']:,.0f}/worker × {num_replicas}R)")

    return pd.DataFrame(rows)


# =========================================================================
# Display helpers
# =========================================================================
def print_budget_summary(result_df: pd.DataFrame, total_gpus: int) -> None:
    if result_df.empty:
        print(f"\n  No results for {total_gpus} GPU budget.")
        return

    print(f"\n{'='*130}")
    print(f"  REPLICA-AWARE COMPARISON — {total_gpus} GPU budget,  {MODEL_PATH}")
    print(f"{'='*130}")

    # Sort by total throughput descending
    display = result_df.sort_values("tokens/s (total)", ascending=False).reset_index(drop=True)
    display_cols = [
        "mode", "worker_gpus", "num_replicas",
        "tp", "dp", "moe_tp", "moe_ep",
        "attn_gpus/worker", "ffn_gpus/worker",
        "bs_per_worker", "global_bs_total",
        "tokens/s (worker)", "tokens/s (total)", "tokens/s/gpu",
        "tokens/s/user", "tpot", "ttft", "memory",
    ]
    avail = [c for c in display_cols if c in display.columns]
    print(display[avail].to_string(index=True))

    # Highlight winner
    winner = display.iloc[0]
    print(f"\n  >>> WINNER: {winner['mode']} — "
          f"{int(winner['worker_gpus'])} GPUs/worker × {int(winner['num_replicas'])} replicas  "
          f"= {winner['tokens/s (total)']:,.0f} tokens/s total "
          f"({winner['tokens/s/gpu']:,.0f} tokens/s/gpu)")

    if winner["mode"] == "AFD":
        print(f"      Config: tp={int(winner['tp'])} dp={int(winner['dp'])} "
              f"moe_tp={int(winner['moe_tp'])} moe_ep={int(winner['moe_ep'])} "
              f"({int(winner['attn_gpus/worker'])}A + {int(winner['ffn_gpus/worker'])}F per worker)")
    else:
        print(f"      Config: tp={int(winner['tp'])} dp={int(winner['dp'])} "
              f"moe_tp={int(winner['moe_tp'])} moe_ep={int(winner['moe_ep'])}")

    # Also show best standard vs best AFD comparison
    for m in ["Standard", "AFD"]:
        sub = display[display["mode"] == m]
        if not sub.empty:
            best = sub.iloc[0]
            print(f"  Best {m:8s}: {best['tokens/s (total)']:>10,.0f} tok/s total  "
                  f"({int(best['worker_gpus'])}gpu × {int(best['num_replicas'])}R, "
                  f"bs={int(best['bs_per_worker'])}, "
                  f"tpot={best['tpot']:.1f}ms)")


# =========================================================================
# Main
# =========================================================================
def main():
    print(f"Replica-Aware AFD Study: {MODEL_PATH} on {SYSTEM}")
    print(f"  ISL={ISL}, OSL={OSL}, TTFT={TTFT}ms, Backend={BACKEND} {VERSION}")
    print(f"  GPU budgets: {GPU_BUDGETS}")
    print(f"  Candidate worker sizes: {CANDIDATE_WORKER_SIZES}")

    database = get_database()

    all_results: list[pd.DataFrame] = []

    for total_gpus in GPU_BUDGETS:
        print(f"\n{'#'*130}")
        print(f"  GPU BUDGET: {total_gpus}")
        print(f"{'#'*130}")

        worker_sizes = _divisors(total_gpus, CANDIDATE_WORKER_SIZES)
        print(f"  Valid worker sizes: {worker_sizes}")

        result_df = analyse_budget(database, total_gpus)
        print_budget_summary(result_df, total_gpus)

        if not result_df.empty:
            fname = f"replica_study_{total_gpus}gpu.csv"
            result_df.to_csv(fname, index=False)
            print(f"\n  Saved: {fname} ({len(result_df)} rows)")
            all_results.append(result_df)

    # Final cross-budget summary
    if all_results:
        print(f"\n\n{'#'*130}")
        print(f"  CROSS-BUDGET SUMMARY")
        print(f"{'#'*130}")

        combined = pd.concat(all_results, ignore_index=True)
        combined_fname = "replica_study_all.csv"
        combined.to_csv(combined_fname, index=False)
        print(f"  Saved combined CSV: {combined_fname}")

        # One-line summary per budget
        for total_gpus in GPU_BUDGETS:
            sub = combined[combined["total_gpus"] == total_gpus]
            if sub.empty:
                print(f"  {total_gpus:3d} GPUs: no results")
                continue
            winner = sub.loc[sub["tokens/s (total)"].idxmax()]
            print(f"  {total_gpus:3d} GPUs: {winner['tokens/s (total)']:>10,.0f} tok/s  "
                  f"({winner['mode']:8s} {int(winner['worker_gpus'])}gpu × "
                  f"{int(winner['num_replicas'])}R, bs={int(winner['bs_per_worker'])})")


if __name__ == "__main__":
    main()
