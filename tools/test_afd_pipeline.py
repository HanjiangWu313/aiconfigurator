#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
4-stage AFD inter-layer pipeline speedup test.

Models the MegaScale-Infer style pipeline where attention and FFN run on
separate GPU groups and communicate via P2P dispatches:

  Stage 1: Attention compute        (on attn GPUs: tp x dp)
  Stage 2: Comm attn → FFN          (moe_pre_dispatch)
  Stage 3: FFN / MoE compute        (on FFN GPUs: moe_tp x moe_ep)
  Stage 4: Comm FFN → attn          (moe_post_dispatch)

The backend owns the pipelined latency model.  This script compares the
backend-returned M=1 and M=max_m totals directly, then prints the M=1 stage
balance to make the bottleneck visible.
"""

from __future__ import annotations

import argparse
import copy
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import BaseModel, get_model
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path

logging.disable(logging.NOTSET)

MODEL = "Qwen/Qwen3-30B-A3B"
SYSTEM = "b200_sxm"
BACKEND = "trtllm"
VERSION = "1.2.0rc5"
ISL, OSL = 4000, 500

STAGE_NAMES = ["Attn compute", "Comm A→F", "FFN compute", "Comm F→A"]


def classify(model: BaseModel, op: str) -> tuple[int, str]:
    if model._is_comm_a2f_op(op):
        return 1, STAGE_NAMES[1]
    if model._is_comm_f2a_op(op):
        return 3, STAGE_NAMES[3]
    if model._is_attn_op(op):
        return 0, STAGE_NAMES[0]
    return 2, STAGE_NAMES[2]


def stage_times(model: BaseModel, breakdown: dict[str, float]) -> list[float]:
    buckets = [0.0] * 4
    for op, lat in breakdown.items():
        buckets[classify(model, op)[0]] += lat
    return buckets


def run(mc, bs):
    db = PerfDatabase(SYSTEM, BACKEND, VERSION, systems_dir=str(get_system_config_path()))
    backend = get_backend(BACKEND)
    model = get_model(MODEL, mc, BACKEND)
    sess = InferenceSession(model=model, database=db, backend=backend)
    rc = config.RuntimeConfig(isl=ISL, osl=OSL, prefix=0, ttft=600.0, tpot=100.0, batch_size=bs)
    ctx = sess.run_static(mode="static_ctx", runtime_config=copy.deepcopy(rc))
    gen = sess.run_static(mode="static_gen", runtime_config=copy.deepcopy(rc))
    return model, ctx.get_context_latency_dict(), gen.get_generation_latency_dict()


def mc(tp=1, dp=1, mtp=1, ep=4, microbatches=1):
    return config.ModelConfig(
        gemm_quant_mode=common.GEMMQuantMode.fp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8,
        comm_quant_mode=common.CommQuantMode.half,
        enable_afd=True,
        tp_size=tp,
        attention_dp_size=dp,
        moe_tp_size=mtp,
        moe_ep_size=ep,
        afd_num_microbatches=microbatches,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, choices=(1, 4), default=4)
    p.add_argument("-b", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--moe-tp", type=int, default=1)
    p.add_argument("--moe-ep", type=int, default=4)
    a = p.parse_args()
    max_m = a.m

    attn_gpus = a.tp * a.dp
    ffn_gpus = a.moe_tp * a.moe_ep
    total_gpus = attn_gpus + ffn_gpus

    print(f"\n{'=' * 70}")
    print("  AFD 4-Stage Pipeline Speedup Test")
    print(f"  Model: {MODEL}  System: {SYSTEM}  Backend: {BACKEND} {VERSION}")
    print(f"  ISL={ISL}  OSL={OSL}  BS={a.b}")
    print(f"  Attn GPUs: {attn_gpus} (tp={a.tp} x dp={a.dp})")
    print(f"  FFN  GPUs: {ffn_gpus} (moe_tp={a.moe_tp} x moe_ep={a.moe_ep})")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Speedup: backend M=1 total / backend M={max_m} total")
    print(f"{'=' * 70}")

    # Baseline (M=1) and pipelined (M=max_m)
    baseline_model, ctx_b, gen_b = run(mc(a.tp, a.dp, a.moe_tp, a.moe_ep, 1), a.b)
    _, ctx_p, gen_p = run(mc(a.tp, a.dp, a.moe_tp, a.moe_ep, max_m), a.b)
    ctx_s = stage_times(baseline_model, ctx_b)
    gen_s = stage_times(baseline_model, gen_b)

    def print_breakdown(label, breakdown_m1, breakdown_mx, stages, microbatches):
        total = sum(stages)
        bottleneck = max(stages)
        microbatch_header = f"M={microbatches} (ms)"

        # Per-op breakdown: M=1 and M=max_m side by side
        print(f"\n  {label} op breakdown:")
        print(f"  {'Op':<30} {'Stage':<16} {'M=1 (ms)':>10}  {microbatch_header:>10}  {'%':>5}")
        print(f"  {'-' * 77}")
        for op, lat in sorted(breakdown_m1.items(), key=lambda x: -x[1]):
            if lat == 0:
                continue
            _, sname = classify(baseline_model, op)
            lat_p = breakdown_mx.get(op, 0.0)
            pct = lat / total * 100
            print(f"  {op:<30} {sname:<16} {lat:>10.3f}  {lat_p:>10.3f}  {pct:>5.1f}%")
        total_p = sum(breakdown_mx.values())
        print(f"  {'─' * 77}")
        print(f"  {'Total':<46} {total:>10.3f}  {total_p:>10.3f}  100.0%")

        direct_speedup = total / total_p if total_p > 0 else float("inf")

        # Stage summary + direct speedup
        print(f"\n  {label} stages (M=1):  " + "  ".join(f"{STAGE_NAMES[i]}={stages[i]:.3f}ms" for i in range(4)))
        print(f"  bottleneck={bottleneck:.3f}ms  stage_balance={total / bottleneck:.2f}x")
        print(f"  Direct M=1 → M={microbatches}: {total:.3f}ms → {total_p:.3f}ms  speedup={direct_speedup:.2f}x")

    print_breakdown("Prefill", ctx_b, ctx_p, ctx_s, max_m)
    print_breakdown("Decode", gen_b, gen_p, gen_s, max_m)

    print()


if __name__ == "__main__":
    main()
