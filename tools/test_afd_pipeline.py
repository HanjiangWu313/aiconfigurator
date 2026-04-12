#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
4-stage AFD inter-layer pipeline speedup test.

Models the MegaScale-Infer style pipeline where attention and FFN run on
separate GPU groups and communicate via P2P dispatches:

  Stage 1: Attention compute        (on attn GPUs: tp × dp)
  Stage 2: Comm attn → FFN          (moe_pre_dispatch)
  Stage 3: FFN / MoE compute        (on FFN GPUs: moe_tp × moe_ep)
  Stage 4: Comm FFN → attn          (moe_post_dispatch)

With M microbatches (M ≤ 4) flowing through the pipeline, the analytical
model for pipelined latency is:

  t_pipe = t_sum / M  +  (M-1)/M × max(t_s1, t_s2, t_s3, t_s4)

At M=1 this equals the sequential sum (no overlap).
As M→∞ it converges to the bottleneck stage latency.
"""
from __future__ import annotations
import argparse, copy, logging, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path

logging.disable(logging.NOTSET)

MODEL = "Qwen/Qwen3-30B-A3B"
SYSTEM = "b200_sxm"
BACKEND = "trtllm"
VERSION = "1.2.0rc5"
ISL, OSL = 4000, 500

_ATTN = {"embedding", "add_norm_1", "qkv_gemm", "attention", "proj_gemm", "ar_1", "p2p"}
_A2F = {"moe_pre_dispatch"}
_F2A = {"moe_post_dispatch"}
STAGE_NAMES = ["Attn compute", "Comm A→F", "FFN compute", "Comm F→A"]


def classify(op: str) -> tuple[int, str]:
    base = op.replace("context_", "").replace("generation_", "")
    if base in _A2F:   return 1, STAGE_NAMES[1]
    elif base in _F2A: return 3, STAGE_NAMES[3]
    elif base in _ATTN: return 0, STAGE_NAMES[0]
    else:               return 2, STAGE_NAMES[2]


def stage_times(breakdown: dict[str, float]) -> list[float]:
    buckets = [0.0] * 4
    for op, lat in breakdown.items():
        buckets[classify(op)[0]] += lat
    return buckets


def run(mc, bs):
    db = PerfDatabase(SYSTEM, BACKEND, VERSION, systems_dir=str(get_system_config_path()))
    backend = get_backend(BACKEND)
    model = get_model(MODEL, mc, BACKEND)
    sess = InferenceSession(model=model, database=db, backend=backend)
    rc = config.RuntimeConfig(isl=ISL, osl=OSL, prefix=0, ttft=600.0, tpot=100.0, batch_size=bs)
    ctx = sess.run_static(mode="static_ctx", runtime_config=copy.deepcopy(rc))
    gen = sess.run_static(mode="static_gen", runtime_config=copy.deepcopy(rc))
    return ctx.get_context_latency_dict(), gen.get_generation_latency_dict()


def mc(tp=1, dp=1, mtp=1, ep=4, M=1):
    return config.ModelConfig(
        gemm_quant_mode=common.GEMMQuantMode.fp8, kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8, moe_quant_mode=common.MoEQuantMode.fp8,
        comm_quant_mode=common.CommQuantMode.half, enable_afd=True,
        tp_size=tp, attention_dp_size=dp, moe_tp_size=mtp, moe_ep_size=ep, afd_num_microbatches=M)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=4)
    p.add_argument("-b", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--moe-tp", type=int, default=1)
    p.add_argument("--moe-ep", type=int, default=4)
    a = p.parse_args()
    max_m = min(a.m, 4)

    attn_gpus = a.tp * a.dp
    ffn_gpus = a.moe_tp * a.moe_ep
    total_gpus = attn_gpus + ffn_gpus

    print(f"\n{'='*70}")
    print(f"  AFD 4-Stage Pipeline Speedup Test")
    print(f"  Model: {MODEL}  System: {SYSTEM}  Backend: {BACKEND} {VERSION}")
    print(f"  ISL={ISL}  OSL={OSL}  BS={a.b}")
    print(f"  Attn GPUs: {attn_gpus} (tp={a.tp} × dp={a.dp})")
    print(f"  FFN  GPUs: {ffn_gpus} (moe_tp={a.moe_tp} × moe_ep={a.moe_ep})")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Formula: t_pipe = t_sum/M + (M-1)/M × max(stages), M ≤ 4")
    print(f"{'='*70}")

    # Baseline (M=1) and pipelined (M=max_m)
    ctx_b, gen_b = run(mc(a.tp, a.dp, a.moe_tp, a.moe_ep, 1), a.b)
    ctx_p, gen_p = run(mc(a.tp, a.dp, a.moe_tp, a.moe_ep, max_m), a.b)
    ctx_s = stage_times(ctx_b)
    gen_s = stage_times(gen_b)

    def print_breakdown(label, breakdown_m1, breakdown_mx, stages, M):
        total = sum(stages)
        bottleneck = max(stages)

        # Per-op breakdown: M=1 and M=max_m side by side
        print(f"\n  {label} op breakdown:")
        print(f"  {'Op':<30} {'Stage':<16} {'M=1 (ms)':>10}  {'M={} (ms)'.format(M):>10}  {'%':>5}")
        print(f"  {'-'*77}")
        for op, lat in sorted(breakdown_m1.items(), key=lambda x: -x[1]):
            if lat == 0:
                continue
            _, sname = classify(op)
            lat_p = breakdown_mx.get(op, 0.0)
            pct = lat / total * 100
            print(f"  {op:<30} {sname:<16} {lat:>10.3f}  {lat_p:>10.3f}  {pct:>5.1f}%")
        total_p = sum(breakdown_mx.values())
        print(f"  {'─'*77}")
        print(f"  {'Total':<46} {total:>10.3f}  {total_p:>10.3f}  100.0%")

        # Stage summary + speedup table
        print(f"\n  {label} stages (M=1):  " + "  ".join(f"{STAGE_NAMES[i]}={stages[i]:.3f}ms" for i in range(4)))
        print(f"  bottleneck={bottleneck:.3f}ms  max_speedup={total/bottleneck:.2f}x")
        print(f"  {'M':>3}  {'Time (ms)':>10}  {'Speedup':>8}")
        print(f"  {'1':>3}  {total:>10.3f}  {'1.00x':>8}")
        for m in range(2, M + 1):
            t = total / m + (m - 1) / m * bottleneck
            print(f"  {m:>3}  {t:>10.3f}  {total/t:>.2f}x")

    print_breakdown("Prefill", ctx_b, ctx_p, ctx_s, max_m)
    print_breakdown("Decode",  gen_b, gen_p, gen_s, max_m)

    print()


if __name__ == "__main__":
    main()
