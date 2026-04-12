#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Disaggregated + AFD study for Qwen3-30B-A3B.

Compares three deployment modes at each GPU budget:
  1. Standard disagg   — prefill/decode workers, shared attention+MoE GPUs
  2. Disagg + AFD      — prefill/decode workers, with AFD (attention and MoE
                         on decoupled GPU groups within each worker)
  3. (reference) Agg   — aggregated (no P/D split), from the agg study

Usage:
    cd /scratch1/hanjiang/aiconfigurator
    source aiconfigvenv/bin/activate
    python tools/afd_disagg_study.py
    python tools/afd_disagg_study.py --breakdown --gpus 8 16

Key dimensions (per worker type — prefill and decode independently):
  - tp, dp, moe_tp, moe_ep
  - num_gpu_per_worker = tp*dp  (standard) or tp*dp + moe_tp*moe_ep (AFD)
  - replica GPUs = prefill_workers * pf_gpus + decode_workers * dec_gpus

Model : Qwen/Qwen3-30B-A3B  (MoE, 128 experts, topk=8)
System: b200_sxm  |  Backend: trtllm 1.2.0rc5
ISL=4000, OSL=500, fp8 quantization
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
logging.disable(logging.WARNING)

# ---- Project imports -----------------------------------------------------
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.pareto_analysis import (
    disagg_pareto,
    disagg_afd_pareto,
    get_pareto_front,
)
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.utils import enumerate_parallel_config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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

GPU_BUDGETS = [8, 16, 32]

# Disagg-specific tuning
PREFILL_LATENCY_CORRECTION = 1.1
DECODE_LATENCY_CORRECTION = 1.08
PREFILL_MAX_BS = 1
DECODE_MAX_BS = 512


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
# 1. Standard disagg (no AFD)
# =========================================================================
def run_disagg_standard(database: PerfDatabase, total_gpus: int) -> pd.DataFrame:
    """Run disagg_pareto with standard (non-AFD) workers."""
    print(f"\n{'='*90}")
    print(f"  DISAGG STANDARD — {total_gpus} GPUs")
    print(f"{'='*90}")

    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    # For trtllm MoE: tp*dp == moe_tp*moe_ep, worker_gpus = tp*dp
    # Restrict to moe_tp=1 (best for this model) and tp=[1,2] to keep search fast
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
        print("  No valid disagg parallel configs found.")
        return pd.DataFrame()

    print(f"  Prefill configs: {len(prefill_configs)},  Decode configs: {len(decode_configs)}")

    # Replica constraint: total GPUs per replica <= total_gpus
    # Let disagg_pareto handle the combinatorial worker matching
    replica_gpu_list = sorted({g for g in range(2, total_gpus + 1)})

    try:
        result_df = disagg_pareto(
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
            max_num_gpu=total_gpus,
            prefill_max_num_worker=32,
            decode_max_num_worker=32,
        )
        if result_df is not None and not result_df.empty:
            print(f"  Got {len(result_df)} results")
        else:
            print("  No results from disagg_pareto.")
            result_df = pd.DataFrame()
        return result_df
    except Exception as e:
        print(f"  Disagg standard failed: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()


# =========================================================================
# 2. Disagg + AFD
# =========================================================================
def run_disagg_afd(database: PerfDatabase, total_gpus: int) -> pd.DataFrame:
    """Run disagg_afd_pareto with AFD-enabled workers."""
    print(f"\n{'='*90}")
    print(f"  DISAGG + AFD — {total_gpus} GPUs")
    print(f"{'='*90}")

    model_config = get_base_model_config()
    runtime_config = get_runtime_config()

    # For AFD: num_gpu_per_worker = tp*dp + moe_tp*moe_ep (decoupled)
    # Restrict to tp=[1], moe_tp=[1] (best for this model), vary dp and moe_ep
    tp_list = [1, 2]
    dp_list = [1, 2, 4, 8]
    moe_tp_list = [1]
    moe_ep_list = [1, 2, 4, 8]
    # Worker sizes = all valid sums of attn_gpus + ffn_gpus
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
        print("  No valid AFD disagg parallel configs found.")
        return pd.DataFrame()

    print(f"  Prefill AFD configs: {len(prefill_configs)},  Decode AFD configs: {len(decode_configs)}")

    replica_gpu_list = sorted({g for g in range(2, total_gpus + 1)})

    try:
        result_df = disagg_afd_pareto(
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
            max_num_gpu=total_gpus,
            prefill_max_num_worker=32,
            decode_max_num_worker=32,
        )
        if result_df is not None and not result_df.empty:
            print(f"  Got {len(result_df)} results")
        else:
            print("  No results from disagg_afd_pareto.")
            result_df = pd.DataFrame()
        return result_df
    except Exception as e:
        print(f"  Disagg AFD failed: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()


# =========================================================================
# 3. Pretty-print comparison
# =========================================================================
DISAGG_KEY_COLS = [
    "(p)tp", "(p)pp", "(p)dp", "(p)moe_tp", "(p)moe_ep",
    "(p)workers",
    "(p)num_attn_gpus", "(p)num_ffn_gpus",
    "(d)tp", "(d)pp", "(d)dp", "(d)moe_tp", "(d)moe_ep",
    "(d)workers",
    "(d)num_attn_gpus", "(d)num_ffn_gpus",
    "num_total_gpus",
    "(p)bs", "(d)bs",
    "tokens/s/gpu", "tokens/s", "tokens/s/user",
    "seq/s", "seq/s/gpu",
    "ttft", "tpot", "request_latency",
    "(p)memory", "(d)memory",
]


def print_comparison(std_df: pd.DataFrame, afd_df: pd.DataFrame, total_gpus: int) -> None:
    """Print side-by-side comparison of disagg standard vs disagg AFD."""
    print(f"\n{'='*130}")
    print(f"  DISAGG COMPARISON: STANDARD vs AFD  ({total_gpus} GPUs, {MODEL_PATH})")
    print(f"{'='*130}")

    def _print_top(label: str, df: pd.DataFrame, n: int = 15) -> None:
        if df.empty:
            print(f"\n  [{label}] — no results")
            return

        pareto = get_pareto_front(df, "tokens/s/user", "tokens/s/gpu")
        if pareto.empty:
            pareto = df.sort_values("tokens/s/gpu", ascending=False).head(n)
        else:
            pareto = pareto.sort_values("tokens/s/gpu", ascending=False).head(n)

        available = [c for c in DISAGG_KEY_COLS if c in pareto.columns]
        display = pareto[available].reset_index(drop=True)

        print(f"\n  [{label}] Top {len(display)} Pareto configs (sorted by tokens/s/gpu):")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(display.to_string(index=True))
        print()

    _print_top("DISAGG STANDARD", std_df)
    _print_top("DISAGG + AFD", afd_df)

    # Best single-config comparison
    if not std_df.empty and not afd_df.empty:
        best_std = std_df.loc[std_df["tokens/s/gpu"].idxmax()]
        best_afd = afd_df.loc[afd_df["tokens/s/gpu"].idxmax()]
        print(f"  {'='*90}")
        print(f"  BEST tokens/s/gpu ({total_gpus} GPUs, disagg):")
        _prt_best("STANDARD", best_std)
        _prt_best("AFD     ", best_afd)
        speedup = best_afd["tokens/s/gpu"] / best_std["tokens/s/gpu"] if best_std["tokens/s/gpu"] > 0 else float("inf")
        print(f"    AFD speedup: {speedup:.2f}x")
    elif not std_df.empty:
        best_std = std_df.loc[std_df["tokens/s/gpu"].idxmax()]
        print(f"  BEST tokens/s/gpu ({total_gpus} GPUs, disagg standard only):")
        _prt_best("STANDARD", best_std)
    elif not afd_df.empty:
        best_afd = afd_df.loc[afd_df["tokens/s/gpu"].idxmax()]
        print(f"  BEST tokens/s/gpu ({total_gpus} GPUs, disagg AFD only):")
        _prt_best("AFD", best_afd)


def _prt_best(label: str, row: pd.Series) -> None:
    """Helper: print one best-result summary line for disagg."""
    p_tp = int(row.get("(p)tp", 0))
    p_dp = int(row.get("(p)dp", 0))
    p_moe_tp = int(row.get("(p)moe_tp", 0))
    p_moe_ep = int(row.get("(p)moe_ep", 0))
    d_tp = int(row.get("(d)tp", 0))
    d_dp = int(row.get("(d)dp", 0))
    d_moe_tp = int(row.get("(d)moe_tp", 0))
    d_moe_ep = int(row.get("(d)moe_ep", 0))
    p_w = int(row.get("(p)workers", 0))
    d_w = int(row.get("(d)workers", 0))
    n_gpu = int(row.get("num_total_gpus", 0))
    tps_gpu = row.get("tokens/s/gpu", 0)
    print(f"    {label}: {tps_gpu:.2f} tok/s/gpu  "
          f"PF[tp{p_tp}dp{p_dp}ep{p_moe_ep}×{p_w}w] "
          f"DEC[tp{d_tp}dp{d_dp}ep{d_moe_ep}×{d_w}w] "
          f"total={n_gpu}gpu")


# =========================================================================
# 4. Kernel-level breakdown helpers
# =========================================================================
def build_model_config_from_disagg_row(
    row: pd.Series, phase: str, is_afd: bool = False,
) -> config.ModelConfig:
    """Reconstruct a ModelConfig from a disagg pareto row for a given phase.

    Args:
        row: a Series from the disagg Pareto DataFrame.
        phase: "(p)" for prefill worker, "(d)" for decode worker.
        is_afd: whether to enable AFD flags.
    """
    mc = get_base_model_config()
    mc.tp_size = int(row[f"{phase}tp"])
    mc.pp_size = int(row[f"{phase}pp"])
    mc.attention_dp_size = int(row[f"{phase}dp"])
    mc.moe_tp_size = int(row[f"{phase}moe_tp"])
    mc.moe_ep_size = int(row[f"{phase}moe_ep"])
    if is_afd:
        mc.enable_afd = True
        mc.num_attn_gpus = int(row.get(f"{phase}num_attn_gpus", mc.tp_size * mc.attention_dp_size))
        mc.num_ffn_gpus = int(row.get(f"{phase}num_ffn_gpus", mc.moe_tp_size * mc.moe_ep_size))
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


def run_breakdown_for_disagg_best(
    std_df: pd.DataFrame, afd_df: pd.DataFrame,
    total_gpus: int, database: PerfDatabase,
) -> None:
    """Run kernel breakdown for the best disagg Standard and AFD configs."""
    for mode_label, df, is_afd in [
        ("DISAGG-STD", std_df, False),
        ("DISAGG-AFD", afd_df, True),
    ]:
        if df.empty:
            continue
        best = df.loc[df["tokens/s/gpu"].idxmax()]

        for phase, phase_label in [("(p)", "PREFILL"), ("(d)", "DECODE")]:
            tp = int(best[f"{phase}tp"])
            dp = int(best[f"{phase}dp"])
            moe_tp = int(best[f"{phase}moe_tp"])
            moe_ep = int(best[f"{phase}moe_ep"])
            mc = build_model_config_from_disagg_row(best, phase, is_afd=is_afd)

            # Use prefill bs or decode bs depending on phase
            bs_col = f"{phase}bs"
            bs = int(best.get(bs_col, 1))

            if is_afd:
                tag = (f"tp={tp} dp={dp} moe_tp={moe_tp} moe_ep={moe_ep}"
                       f" ({mc.num_attn_gpus}A+{mc.num_ffn_gpus}F)")
            else:
                tag = f"tp={tp} dp={dp} moe_tp={moe_tp} moe_ep={moe_ep}"

            label = f"{total_gpus}G {mode_label} {phase_label}: {tag}"
            print_breakdown(label, mc, bs, ISL, database)


# =========================================================================
# 5. Main
# =========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disaggregated + AFD study for Qwen3-30B-A3B.",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="After finding the best config per budget, print kernel-level "
             "latency breakdowns for prefill and decode workers.",
    )
    parser.add_argument(
        "--gpus", type=int, nargs="*", default=None,
        help=f"GPU budgets to sweep (default: {GPU_BUDGETS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    budgets = args.gpus if args.gpus else GPU_BUDGETS

    print(f"Disagg + AFD Study: {MODEL_PATH} on {SYSTEM}")
    print(f"  ISL={ISL}, OSL={OSL}, TTFT={TTFT}ms, Backend={BACKEND} {VERSION}")
    print(f"  Prefill correction={PREFILL_LATENCY_CORRECTION}, "
          f"Decode correction={DECODE_LATENCY_CORRECTION}")
    if args.breakdown:
        print("  Kernel breakdown: ENABLED")

    database = get_database()

    for total_gpus in budgets:
        print(f"\n{'#'*120}")
        print(f"  GPU BUDGET: {total_gpus}")
        print(f"{'#'*120}")

        std_df = run_disagg_standard(database, total_gpus)
        afd_df = run_disagg_afd(database, total_gpus)
        print_comparison(std_df, afd_df, total_gpus)

        # Save per-budget CSVs
        if not std_df.empty:
            fname = f"disagg_study_standard_{total_gpus}gpu.csv"
            std_df.to_csv(fname, index=False)
            print(f"  Saved: {fname} ({len(std_df)} rows)")
        if not afd_df.empty:
            fname = f"disagg_study_afd_{total_gpus}gpu.csv"
            afd_df.to_csv(fname, index=False)
            print(f"  Saved: {fname} ({len(afd_df)} rows)")

        # Kernel-level breakdown
        if args.breakdown:
            run_breakdown_for_disagg_best(std_df, afd_df, total_gpus, database)


if __name__ == "__main__":
    main()
