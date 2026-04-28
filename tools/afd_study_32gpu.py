#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD (Attention-FFN Disaggregation) study for Qwen3-30B-A3B on 32 GPUs.

This script enumerates all valid AFD parallel configurations for a fixed
GPU budget, runs ``agg_pareto`` for each, and produces a comparison table
showing how different attn:FFN GPU ratios affect throughput and latency.

Usage:
    cd /scratch1/hanjiang/aiconfigurator
    source aiconfigvenv/bin/activate
    python tools/afd_study_32gpu.py

The study answers: given 32 B200 GPUs serving Qwen3-30B-A3B (MoE, 128
experts, topk=8), what is the best (tp, dp, moe_tp, moe_ep) configuration
when attention and FFN GPUs are physically separated (AFD)?

Key sweep dimensions:
  - num_attn_gpus = tp × dp   (attention group)
  - num_ffn_gpus  = moe_tp × moe_ep  (FFN/MoE group)
  - num_attn_gpus + num_ffn_gpus = TOTAL_GPUS
  - Batch size is swept *internally* by agg_pareto / find_best_agg_result_under_constraints

For comparison the script also runs the standard (non-AFD) agg_pareto
where all GPUs share the same model.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import warnings

import pandas as pd

# ---- Silence noisy warnings during the sweep ----------------------------
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)  # Suppress all WARNING/INFO from sub-modules

# ---- Project imports -----------------------------------------------------
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.pareto_analysis import agg_pareto, agg_afd_pareto, get_pareto_front
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.utils import enumerate_parallel_config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# =========================================================================
# Configuration — adjust these to explore different setups
# =========================================================================
MODEL_PATH = "Qwen/Qwen3-30B-A3B"
SYSTEM = "b200_sxm"
BACKEND = "trtllm"
VERSION = "1.2.0rc5"

ISL = 4000       # input sequence length
OSL = 500        # output sequence length
PREFIX = 0
TTFT = 600.0     # target TTFT (ms)
TPOT_LIST = list(range(1, 20, 1)) + list(range(20, 300, 5))

# Study multiple GPU budgets to see scaling behaviour
GPU_BUDGETS = [8, 16, 32]


def get_database() -> PerfDatabase:
    systems_dir = get_system_config_path()
    return PerfDatabase(SYSTEM, BACKEND, VERSION, systems_dir=str(systems_dir))


def get_base_model_config() -> config.ModelConfig:
    """Return a base ModelConfig with fp8 quant modes (typical for B200)."""
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
# 1. Standard (non-AFD) agg pareto — all GPUs share the full model
# =========================================================================
def run_standard_pareto(database: PerfDatabase, total_gpus: int) -> pd.DataFrame:
    """Run standard agg_pareto with shared GPUs (no AFD)."""
    print(f"\n{'='*90}")
    print(f"  STANDARD AGG PARETO (non-AFD) — {total_gpus} GPUs")
    print(f"{'='*90}")

    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    # Standard parallel config enumeration (tp*dp == moe_tp*moe_ep)
    max_dim = min(total_gpus, 32)
    dim_list = sorted({2**i for i in range(6) if 2**i <= max_dim} | {total_gpus})
    parallel_configs = enumerate_parallel_config(
        num_gpu_list=[total_gpus],
        tp_list=[1, 2, 4, 8],
        pp_list=[1],
        dp_list=dim_list,
        moe_tp_list=[1],
        moe_ep_list=dim_list,
        is_moe=True,
        backend=common.BackendName.trtllm,
    )

    if not parallel_configs:
        print("  No valid standard parallel configs found.")
        return pd.DataFrame()

    print(f"  {len(parallel_configs)} configs:", end="")
    for cfg in parallel_configs:
        tp, pp, dp, moe_tp, moe_ep = cfg
        print(f"  tp{tp}dp{dp}ep{moe_ep}", end="")
    print()

    try:
        result_df = agg_pareto(
            model_path=MODEL_PATH,
            runtime_config=runtime_config,
            database=copy.deepcopy(database),
            backend_name=BACKEND,
            model_config=model_config,
            parallel_config_list=parallel_configs,
        )
        return result_df
    except Exception as e:
        print(f"  Standard pareto failed: {e}")
        return pd.DataFrame()


# =========================================================================
# 2. AFD agg pareto — attention and FFN on separate GPU groups
# =========================================================================
def run_afd_pareto(database: PerfDatabase, total_gpus: int) -> pd.DataFrame:
    """Run agg_afd_pareto with decoupled attn/FFN GPUs."""
    print(f"\n{'='*90}")
    print(f"  AFD AGG PARETO — {total_gpus} GPUs  (attn_gpus + ffn_gpus = {total_gpus})")
    print(f"{'='*90}")

    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    # For AFD: total = tp*dp + moe_tp*moe_ep
    # Focus on power-of-2 splits: (1+N-1, 2+N-2, 4+N-4, 8+N-8, ...)
    all_dims = sorted({2**i for i in range(6)} | {total_gpus})
    # Also add complement values
    extra = set()
    for v in all_dims:
        c = total_gpus - v
        if c > 0:
            extra.add(c)
    all_dims = sorted(set(all_dims) | extra)

    parallel_configs = enumerate_parallel_config(
        num_gpu_list=[total_gpus],
        tp_list=[1, 2, 4, 8],
        pp_list=[1],
        dp_list=all_dims,
        moe_tp_list=[1, 2, 4, 8],
        moe_ep_list=all_dims,
        is_moe=True,
        backend=common.BackendName.trtllm,
        enable_afd=True,
    )

    if not parallel_configs:
        print("  No valid AFD parallel configs found.")
        return pd.DataFrame()

    print(f"  {len(parallel_configs)} configs:")
    for cfg in parallel_configs:
        tp, pp, dp, moe_tp, moe_ep = cfg
        n_attn = tp * dp
        n_ffn = moe_tp * moe_ep
        print(f"    tp={tp} dp={dp} moe_tp={moe_tp} moe_ep={moe_ep}"
              f"  →  {n_attn}A + {n_ffn}F = {n_attn + n_ffn}")

    try:
        result_df = agg_afd_pareto(
            model_path=MODEL_PATH,
            runtime_config=runtime_config,
            database=copy.deepcopy(database),
            backend_name=BACKEND,
            model_config=model_config,
            parallel_config_list=parallel_configs,
        )
        return result_df
    except Exception as e:
        print(f"  AFD pareto failed: {e}")
        return pd.DataFrame()


# =========================================================================
# 3. Kernel-level breakdown helpers
# =========================================================================
def build_model_config_from_row(row: pd.Series, is_afd: bool = False) -> config.ModelConfig:
    """Reconstruct a ModelConfig from a pareto-result DataFrame row."""
    mc = get_base_model_config()
    mc.tp_size = int(row["tp"])
    mc.pp_size = int(row["pp"])
    mc.attention_dp_size = int(row["dp"])
    mc.moe_tp_size = int(row["moe_tp"])
    mc.moe_ep_size = int(row["moe_ep"])
    if is_afd:
        mc.enable_afd = True
        mc.num_attn_gpus = int(row.get("num_attn_gpus", mc.tp_size * mc.attention_dp_size))
        mc.num_ffn_gpus = int(row.get("num_ffn_gpus", mc.moe_tp_size * mc.moe_ep_size))
    return mc


def print_breakdown(label: str, model_config: config.ModelConfig,
                    bs: int, ctx_tokens: int, database: PerfDatabase) -> None:
    """Print per-kernel latency breakdown for context and generation phases."""
    print(f"\n{'=' * 90}")
    print(f"  KERNEL BREAKDOWN: {label}")
    print(f"  bs={bs}, ctx_tokens={ctx_tokens}, ISL={ISL}, OSL={OSL}")
    print(f"{'=' * 90}")

    model = get_model(model_path=MODEL_PATH, model_config=model_config, backend_name=BACKEND)
    backend = get_backend(BACKEND)

    # ---- Context (prefill) breakdown ----
    s1 = backend.run_static(
        model, copy.deepcopy(database),
        config.RuntimeConfig(batch_size=1, beam_width=1, isl=ctx_tokens, osl=1, prefix=PREFIX),
        mode="static_ctx",
    )
    ctx_lat = dict(s1.get_context_latency_dict())

    num_ctx_reqs = max(1, math.ceil(ctx_tokens / ISL))
    s2 = backend.run_static(
        model, copy.deepcopy(database),
        config.RuntimeConfig(batch_size=num_ctx_reqs, beam_width=1, isl=ISL, osl=1, prefix=PREFIX),
        mode="static_ctx",
    )
    ctx_attn_lat = s2.get_context_latency_dict()
    scale_factor = max(1, math.ceil(ISL / ctx_tokens))
    ctx_lat["context_attention"] = ctx_attn_lat.get("context_attention", 0) / scale_factor

    total_ctx = sum(ctx_lat.values())
    print(f"\n  CTX (prefill) — one mixed-step latency breakdown:")
    print(f"  {'Op':<35s} {'ms':>10s} {'%':>8s}")
    print(f"  {'-' * 53}")
    for op_name, lat in sorted(ctx_lat.items(), key=lambda x: -x[1]):
        pct = lat / total_ctx * 100 if total_ctx > 0 else 0
        print(f"  {op_name:<35s} {lat:>10.3f} {pct:>7.1f}%")
    print(f"  {'TOTAL':<35s} {total_ctx:>10.3f}")

    # ---- Generation (decode) breakdown ----
    gen_tokens = max(1, bs - num_ctx_reqs)
    s3 = backend.run_static(
        model, copy.deepcopy(database),
        config.RuntimeConfig(batch_size=gen_tokens, beam_width=1, isl=ISL + OSL // 2, osl=2),
        mode="static_gen",
    )
    gen_lat = s3.get_generation_latency_dict()

    total_gen = sum(gen_lat.values())
    print(f"\n  GEN (decode) — one gen-only step latency (gen_tokens={gen_tokens}):")
    print(f"  {'Op':<35s} {'ms':>10s} {'%':>8s}")
    print(f"  {'-' * 53}")
    for op_name, lat in sorted(gen_lat.items(), key=lambda x: -x[1]):
        pct = lat / total_gen * 100 if total_gen > 0 else 0
        print(f"  {op_name:<35s} {lat:>10.3f} {pct:>7.1f}%")
    print(f"  {'TOTAL':<35s} {total_gen:>10.3f}")


def run_breakdown_for_best(std_df: pd.DataFrame, afd_df: pd.DataFrame,
                           total_gpus: int, database: PerfDatabase) -> None:
    """Run kernel breakdown for the best Standard and AFD configs of a budget."""
    for label_prefix, df, is_afd in [
        ("STANDARD", std_df, False),
        ("AFD", afd_df, True),
    ]:
        if df.empty:
            continue
        best = df.loc[df["tokens/s/gpu"].idxmax()]
        tp, dp = int(best["tp"]), int(best["dp"])
        moe_tp, moe_ep = int(best["moe_tp"]), int(best["moe_ep"])
        bs = int(best["bs"])
        mc = build_model_config_from_row(best, is_afd=is_afd)

        if is_afd:
            tag = (f"tp={tp} dp={dp} moe_tp={moe_tp} moe_ep={moe_ep}"
                   f" ({mc.num_attn_gpus}A+{mc.num_ffn_gpus}F)")
        else:
            tag = f"tp={tp} dp={dp} moe_tp={moe_tp} moe_ep={moe_ep}"

        label = f"{total_gpus}G {label_prefix}: {tag}"
        print_breakdown(label, mc, bs, ISL, database)


# =========================================================================
# 4. Pretty-print comparison
# =========================================================================
def print_comparison(std_df: pd.DataFrame, afd_df: pd.DataFrame, total_gpus: int) -> None:
    """Print a side-by-side comparison of the best configs."""
    print(f"\n{'='*120}")
    print(f"  COMPARISON: STANDARD vs AFD  ({total_gpus} GPUs, {MODEL_PATH})")
    print(f"{'='*120}")

    KEY_COLS = [
        "tp", "pp", "dp", "moe_tp", "moe_ep",
        "num_total_gpus", "num_attn_gpus", "num_ffn_gpus",
        "bs", "global_bs",
        "tokens/s/gpu", "tokens/s", "tokens/s/user",
        "seq/s", "seq/s/gpu",
        "ttft", "tpot", "request_latency",
        "memory",
    ]

    def _print_top(label: str, df: pd.DataFrame, n: int = 15) -> None:
        if df.empty:
            print(f"\n  [{label}] — no results")
            return

        # Get pareto front on tokens/s/user vs tokens/s/gpu
        pareto = get_pareto_front(df, "tokens/s/user", "tokens/s/gpu")
        if pareto.empty:
            pareto = df.sort_values("tokens/s/gpu", ascending=False).head(n)
        else:
            pareto = pareto.sort_values("tokens/s/gpu", ascending=False).head(n)

        available = [c for c in KEY_COLS if c in pareto.columns]
        display = pareto[available].reset_index(drop=True)

        print(f"\n  [{label}] Top {len(display)} Pareto configs (sorted by tokens/s/gpu):")
        print(display.to_string(index=True))
        print()

    _print_top("STANDARD (shared GPUs)", std_df)
    _print_top("AFD (decoupled attn/FFN GPUs)", afd_df)

    # Best single config comparison
    if not std_df.empty and not afd_df.empty:
        best_std = std_df.loc[std_df["tokens/s/gpu"].idxmax()]
        best_afd = afd_df.loc[afd_df["tokens/s/gpu"].idxmax()]
        print(f"  {'='*90}")
        print(f"  BEST tokens/s/gpu ({total_gpus} GPUs):")
        print(f"    STANDARD: {best_std['tokens/s/gpu']:.2f}  "
              f"(tp={int(best_std['tp'])} dp={int(best_std['dp'])} "
              f"moe_tp={int(best_std['moe_tp'])} moe_ep={int(best_std['moe_ep'])} "
              f"bs={int(best_std['bs'])})")
        print(f"    AFD:      {best_afd['tokens/s/gpu']:.2f}  "
              f"(tp={int(best_afd['tp'])} dp={int(best_afd['dp'])} "
              f"moe_tp={int(best_afd['moe_tp'])} moe_ep={int(best_afd['moe_ep'])} "
              f"attn_gpus={best_afd.get('num_attn_gpus', 'N/A')} "
              f"ffn_gpus={best_afd.get('num_ffn_gpus', 'N/A')} "
              f"bs={int(best_afd['bs'])})")

        speedup = best_afd["tokens/s/gpu"] / best_std["tokens/s/gpu"] if best_std["tokens/s/gpu"] > 0 else float("inf")
        print(f"    AFD speedup: {speedup:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AFD study for Qwen3-30B-A3B: Standard vs AFD across GPU budgets.",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="After finding the best config per budget, print a kernel-level "
             "latency breakdown for each winner.",
    )
    parser.add_argument(
        "--gpus", type=int, nargs="*", default=None,
        help=f"GPU budgets to sweep (default: {GPU_BUDGETS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    budgets = args.gpus if args.gpus else GPU_BUDGETS

    print(f"AFD Study: {MODEL_PATH} on {SYSTEM}")
    print(f"  ISL={ISL}, OSL={OSL}, TTFT={TTFT}ms, Backend={BACKEND} {VERSION}")
    if args.breakdown:
        print("  Kernel breakdown: ENABLED")

    database = get_database()

    for total_gpus in budgets:
        print(f"\n{'#'*120}")
        print(f"  GPU BUDGET: {total_gpus}")
        print(f"{'#'*120}")

        std_df = run_standard_pareto(database, total_gpus)
        afd_df = run_afd_pareto(database, total_gpus)
        print_comparison(std_df, afd_df, total_gpus)

        # Save per-budget CSVs
        if not std_df.empty:
            fname = f"afd_study_standard_{total_gpus}gpu.csv"
            std_df.to_csv(fname, index=False)
            print(f"  Saved: {fname} ({len(std_df)} rows)")
        if not afd_df.empty:
            fname = f"afd_study_afd_{total_gpus}gpu.csv"
            afd_df.to_csv(fname, index=False)
            print(f"  Saved: {fname} ({len(afd_df)} rows)")

        # Kernel-level breakdown for best configs
        if args.breakdown:
            run_breakdown_for_best(std_df, afd_df, total_gpus, database)


if __name__ == "__main__":
    main()
