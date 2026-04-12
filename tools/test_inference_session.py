#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print GPU placement and KV-transfer links for non-AFD and AFD layouts.

This is a lightweight inspection script for ``DisaggInferenceSession._build_gpu_layout``.
It uses the real implementation from ``inference_session.py`` and prints:

1. worker GPU placement
2. AFD-specific attention / FFN partitions
3. the concrete KV-cache transfer plan built by ``AstraSimManager``
4. the network tier each transfer uses (NVLink / IB)

Run:
    ./aiconfigvenv/bin/python tools/test_inference_session.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from types import SimpleNamespace

logging.basicConfig(level=logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiconfigurator.sdk.astrasim_utils import AstraSimManager, NETWORK_SIM_AVAILABLE
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.inference_session import DisaggInferenceSession


H100_SXM_SYSTEM_SPEC = {
    "node": {
        "num_gpus_per_node": 8,
        "intra_node_bw": 450_000_000_000,
        "inter_node_bw": 25_000_000_000,
        "pcie_bw": 64_000_000_000,
        "p2p_latency": 0.000_01,
    },
    "gpu": {
        "mem_bw": 3_350_000_000_000,
        "mem_capacity": 85_899_345_920,
        "float16_tc_flops": 989_000_000_000_000,
    },
}

KV_BYTES_PER_TOKEN = 163_840


def _build_disagg_session(gpu_layout_strategy: str) -> DisaggInferenceSession:
    """Create a minimal session that can call the real layout builder."""
    fake_database = SimpleNamespace(system_spec=H100_SXM_SYSTEM_SPEC)
    fake_backend = SimpleNamespace(name=SimpleNamespace(value="trtllm"))
    fake_astrasim = SimpleNamespace(enabled=False)
    return DisaggInferenceSession(
        prefill_database=fake_database,
        prefill_backend=fake_backend,
        decode_database=fake_database,
        decode_backend=fake_backend,
        gpu_layout_strategy=gpu_layout_strategy,
        astrasim_manager=fake_astrasim,
    )


def _tier_to_link(tier_name: str) -> str:
    if tier_name == "intra-node":
        return "NVLink"
    if tier_name == "intra-rack":
        return "NVSwitch"
    if tier_name in {"inter-node", "inter-rack"}:
        return "InfiniBand"
    return "Flat"


def _fmt_gpus(gpus: list[int]) -> str:
    return "[" + ", ".join(str(g) for g in gpus) + "]"


def _fmt_nodes(gpus: list[int]) -> str:
    return "[" + ", ".join(str(g // H100_SXM_SYSTEM_SPEC["node"]["num_gpus_per_node"]) for g in gpus) + "]"


def _print_worker(prefix: str, worker: dict) -> None:
    print(
        f"  {prefix}[{worker['worker_id']}]: GPUs={_fmt_gpus(worker['gpu_ids'])}  "
        f"nodes={_fmt_nodes(worker['gpu_ids'])}"
    )
    print(
        f"    pp_stages={worker['pp_stages']}  "
        f"first_stage={worker['first_stage_gpus']}  last_stage={worker['last_stage_gpus']}"
    )
    if worker.get("enable_afd", False):
        print(
            f"    attn_gpu_ids={worker['attn_gpu_ids']}  "
            f"ffn_gpu_ids={worker['ffn_gpu_ids']}"
        )
        for dp_rank, stages in enumerate(worker["attn_dp_pp_stages"]):
            print(f"    attn_dp_pp_stages[{dp_rank}]={stages}")
        print(f"    ffn_pp_stages={worker['ffn_pp_stages']}")


def _print_transfer_trace(
    manager: AstraSimManager,
    layout: dict,
    kv_cache_size: int,
    prefill_batch_size: int,
) -> None:
    transfers = manager.build_kv_transfer_plan(
        gpu_layout=layout,
        kv_cache_size=kv_cache_size,
        prefill_batch_size=prefill_batch_size,
    )
    expected_total = kv_cache_size * len(layout["prefill_worker_layouts"])
    actual_total = sum(t["bytes"] for t in transfers)
    used_src = sorted({t["src"] for t in transfers})
    used_dst = sorted({t["dst"] for t in transfers})
    link_summary: Counter[str] = Counter()

    print(f"  Pairing: {layout['prefill_decode_pairing']}")
    print(
        f"  Transfers generated: {len(transfers)}  "
        f"total_bytes={actual_total:,} / expected={expected_total:,}"
    )
    print(f"  Source GPUs used by KV transfer:      {used_src}")
    print(f"  Destination GPUs used by KV transfer: {used_dst}")
    print()

    for transfer in transfers:
        tier_key, local_src, local_dst = manager._classify_tier(
            transfer["src"], transfer["dst"]
        )
        tier_name, group_id = tier_key
        link_name = _tier_to_link(tier_name)
        link_summary[f"{link_name}/{tier_name}"] += 1
        dp_suffix = f"  dp_rank={transfer['dp_rank']}" if "dp_rank" in transfer else ""
        print(
            f"    P[{transfer['p_worker']}] -> D[{transfer['d_worker']}]"
            f"  pp_stage={transfer['pp_stage']}{dp_suffix}"
        )
        print(
            f"      GPU {transfer['src']:>2d} (node {transfer['src'] // 8})"
            f" -> GPU {transfer['dst']:>2d} (node {transfer['dst'] // 8})"
            f"  bytes={transfer['bytes'] / 1e6:>9.3f} MB"
            f"  mode={transfer['mode']}"
        )
        print(
            f"      link={link_name}  tier={tier_name}  "
            f"group={group_id}  local=({local_src}->{local_dst})"
        )

    print()
    if link_summary:
        print("  Link summary:")
        for link_name, count in sorted(link_summary.items()):
            print(f"    {link_name:<24s} {count:>3d} transfers")

    if NETWORK_SIM_AVAILABLE:
        latency_ms = manager.simulate_kv_cache_transfer(
            gpu_layout=layout,
            kv_cache_size=kv_cache_size,
            prefill_batch_size=prefill_batch_size,
        )
        print(f"  AstraSim latency: {latency_ms:.4f} ms")
    else:
        print("  AstraSim latency: skipped (network simulator library not available)")


def show_non_afd_reference(manager: AstraSimManager) -> None:
    print("=" * 90)
    print("REFERENCE: Non-AFD Layout")
    print("=" * 90)
    print("  2P + 2D, TP=2, PP=1, strategy=segregated_by_phase")
    print("  This mirrors the existing non-AFD style: workers, PP stages, and concrete links.")
    print()

    session = _build_disagg_session("segregated_by_phase")
    model_config = ModelConfig(
        tp_size=2,
        pp_size=1,
        moe_tp_size=2,
        moe_ep_size=1,
        attention_dp_size=1,
        enable_afd=False,
    )
    layout = session._build_gpu_layout(
        prefill_model_config=model_config,
        prefill_num_worker=2,
        decode_model_config=model_config,
        decode_num_worker=2,
    )

    for worker in layout["prefill_worker_layouts"]:
        _print_worker("Prefill", worker)
    for worker in layout["decode_worker_layouts"]:
        _print_worker("Decode ", worker)
    print()

    kv_cache_size = 4000 * KV_BYTES_PER_TOKEN
    _print_transfer_trace(manager, layout, kv_cache_size, prefill_batch_size=1)
    print()


def show_afd_worker_anatomy() -> None:
    print("=" * 90)
    print("AFD Worker Anatomy")
    print("=" * 90)
    print("  1 worker, TP=2, PP=2, attention_dp=2, moe_tp=1, moe_ep=4")
    print("  This shows exactly what build_worker_layout returns when AFD is enabled.")
    print()

    session = _build_disagg_session("segregated_by_phase")
    model_config = ModelConfig(
        tp_size=2,
        pp_size=2,
        moe_tp_size=1,
        moe_ep_size=4,
        attention_dp_size=2,
        enable_afd=True,
    )
    layout = session._build_gpu_layout(
        prefill_model_config=model_config,
        prefill_num_worker=1,
        decode_model_config=model_config,
        decode_num_worker=0,
    )

    worker = layout["prefill_worker_layouts"][0]
    _print_worker("Prefill", worker)
    print()
    print(
        "  Notes:\n"
        "    - `pp_stages` is the attention DP-replica-0 path used by KV transfer.\n"
        "    - `attn_dp_pp_stages` keeps the full per-replica attention layout.\n"
        "    - `ffn_pp_stages` shows the FFN/MoE GPUs, which do not send KV cache."
    )
    print()


def show_afd_transfer_trace(manager: AstraSimManager, strategy: str) -> None:
    print("=" * 90)
    print(f"AFD Transfer Trace: strategy={strategy}")
    print("=" * 90)
    print("  2P + 2D, TP=1, PP=1, attention_dp=2, moe_tp=1, moe_ep=2")
    print("  Each worker uses 4 GPUs: 2 attention GPUs + 2 FFN GPUs.")
    print("  Only the attention GPUs should appear in the KV transfer plan.")
    print()

    session = _build_disagg_session(strategy)
    model_config = ModelConfig(
        tp_size=1,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=2,
        attention_dp_size=2,
        enable_afd=True,
    )
    layout = session._build_gpu_layout(
        prefill_model_config=model_config,
        prefill_num_worker=2,
        decode_model_config=model_config,
        decode_num_worker=2,
    )

    for worker in layout["prefill_worker_layouts"]:
        _print_worker("Prefill", worker)
    for worker in layout["decode_worker_layouts"]:
        _print_worker("Decode ", worker)
    print()

    kv_cache_size = 8000 * KV_BYTES_PER_TOKEN + 3
    print(
        f"  KV bytes per prefill worker: {kv_cache_size:,} "
        f"({kv_cache_size / 1e6:.3f} MB)"
    )
    print("  The +3 bytes are intentional so we can see remainder bytes stay preserved.")
    print()
    _print_transfer_trace(manager, layout, kv_cache_size, prefill_batch_size=1)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print GPU placement and KV-transfer links for AFD layouts"
    )
    parser.add_argument(
        "--skip-non-afd",
        action="store_true",
        help="Skip the non-AFD reference section",
    )
    args = parser.parse_args()

    manager = AstraSimManager(system_spec=H100_SXM_SYSTEM_SPEC)

    if not args.skip_non_afd:
        show_non_afd_reference(manager)
    show_afd_worker_anatomy()
    show_afd_transfer_trace(manager, "segregated_by_phase")
    show_afd_transfer_trace(manager, "paired_prefill_decode_per_node")


if __name__ == "__main__":
    main()
