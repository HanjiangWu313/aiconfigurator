#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact AFD kernel/transfer tracker over ISL/OSL cases.

This tool intentionally follows the SDK execution path instead of
recomputing the pipeline by hand:

* agg:    InferenceSession.run_agg(...) -> InferenceSummary
* disagg: DisaggInferenceSession.run_disagg_afd(...) -> phase InferenceSummary objects

The printout mirrors the useful parts of
tests/unit/sdk/models/test_agg_disagg_afd.py::test_returns_all_time_breakdown:
summary metrics, per-kernel latency tables, and AFD dispatch totals.
The performance-database mode is selectable; use SILICON when a fully profiled
baseline is available and HYBRID when analytical fallback is acceptable.

Examples:
    ./aiconfigvenv/bin/python tools/afd_transfer_tracker.py
    ./aiconfigvenv/bin/python tools/afd_transfer_tracker.py --serving both --isls 1000,6400 --osls 100,1000
    ./aiconfigvenv/bin/python tools/afd_transfer_tracker.py --tp 2 --dp 4 --moe-tp 1 --moe-ep 8

See docs/afd_astrasim_validation.md for native build and comparison steps.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from aiconfigurator.sdk import astrasim_utils, common, config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import DisaggInferenceSession, InferenceSession
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel, get_model
from aiconfigurator.sdk.perf_database import (
    PerfDatabase,
    get_latest_database_version,
    get_system_config_path,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
)
logging.getLogger("aiconfigurator").setLevel(logging.WARNING)
logging.getLogger("aiconfigurator.sdk.inference_session").setLevel(logging.WARNING)
logging.getLogger("aiconfigurator.sdk.astrasim_utils").setLevel(logging.DEBUG)
logging.getLogger("aiconfigurator.sdk.utils").setLevel(logging.WARNING)


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_SYSTEM = "b200_sxm"
DEFAULT_BACKEND = "trtllm"
DEFAULT_VERSION = "1.2.0rc5"
DEFAULT_DATABASE_MODE = common.DatabaseMode.HYBRID

SUMMARY_KEYS_AGG = [
    "request_rate",
    "bs",
    "global_bs",
    "num_attn_gpus",
    "num_ffn_gpus",
    "ttft",
    "tpot",
    "request_latency",
    "seq/s",
    "seq/s/gpu",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
]

SUMMARY_KEYS_DISAGG = [
    "request_rate",
    "(p)bs",
    "(p)global_bs",
    "(p)workers",
    "(d)bs",
    "(d)global_bs",
    "(d)workers",
    "ttft",
    "tpot",
    "request_latency",
    "seq/s",
    "seq/s/gpu",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
    "num_total_gpus",
    "kv_network_latency_ms",
    "kv_cache_size_bytes",
]


@dataclass
class AggRunResult:
    summary: InferenceSummary
    context_kernels: InferenceSummary
    generation_kernels: InferenceSummary


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _version_or_latest(system: str, backend: str, version: str | None) -> str:
    if not version or version == "latest":
        return get_latest_database_version(system, backend)
    return version


def _database_mode(args: argparse.Namespace) -> common.DatabaseMode:
    return common.DatabaseMode[args.database_mode]


def _build_database(
    *,
    system: str,
    backend: str,
    version: str | None,
    use_astrasim: bool,
) -> PerfDatabase:
    db = PerfDatabase(
        system=system,
        backend=backend,
        version=_version_or_latest(system, backend, version),
        systems_dir=str(get_system_config_path()),
        use_astrasim=use_astrasim,
    )
    return db


def _require_native_astrasim(use_astrasim: bool) -> None:
    if use_astrasim and not (astrasim_utils.NETWORK_SIM_AVAILABLE and astrasim_utils.NETWORK_SIM_UNAWARE_AVAILABLE):
        raise RuntimeError(
            "--use-astrasim requires both native AstraSim bindings "
            "(NETWORK_SIM_AVAILABLE and NETWORK_SIM_UNAWARE_AVAILABLE). "
            f"Detected NETWORK_SIM_AVAILABLE={astrasim_utils.NETWORK_SIM_AVAILABLE}, "
            "NETWORK_SIM_UNAWARE_AVAILABLE="
            f"{astrasim_utils.NETWORK_SIM_UNAWARE_AVAILABLE}. "
            "Install/build both bindings or pass --no-use-astrasim."
        )


def _num_attn_gpus(args: argparse.Namespace) -> int:
    return args.tp * args.attention_dp


def _num_ffn_gpus(args: argparse.Namespace) -> int:
    return args.moe_tp * args.moe_ep


def _afd_label(args: argparse.Namespace) -> str:
    return f"{_num_attn_gpus(args)}A:{_num_ffn_gpus(args)}F"


def _build_model_config(args: argparse.Namespace) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=args.tp,
        pp_size=args.pp,
        moe_tp_size=args.moe_tp,
        moe_ep_size=args.moe_ep,
        attention_dp_size=args.attention_dp,
        enable_afd=True,
        enable_wideep=args.enable_wideep,
        num_attn_gpus=_num_attn_gpus(args),
        num_ffn_gpus=_num_ffn_gpus(args),
        afd_num_microbatches=args.afd_num_microbatches,
    )


def _runtime_config(args: argparse.Namespace, isl: int, osl: int) -> config.RuntimeConfig:
    return config.RuntimeConfig(
        batch_size=args.batch_size,
        beam_width=args.beam_width,
        isl=isl,
        osl=osl,
        prefix=args.prefix,
    )


def _ctx_tokens_for_case(args: argparse.Namespace, runtime_config: config.RuntimeConfig) -> int:
    return args.ctx_tokens if args.ctx_tokens is not None else runtime_config.isl


def _classify_stage(model: BaseModel, op_name: str) -> str:
    if model._is_comm_a2f_op(op_name):
        return "a2f"
    if model._is_comm_f2a_op(op_name):
        return "f2a"
    if model._is_attn_op(op_name):
        return "attn"
    return "ffn"


def _phase_dict(summary: InferenceSummary, phase: str) -> dict[str, float]:
    if phase == "context":
        return summary.get_context_latency_dict()
    if phase == "generation":
        return summary.get_generation_latency_dict()
    raise ValueError(f"Unknown phase '{phase}'.")


def _fmt_value(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):14.4f}"
    return f"{value!s:>14s}"


def _stage_total(
    model: BaseModel,
    latencies: dict[str, float],
    stage_names: set[str],
) -> float:
    return sum(float(value) for key, value in latencies.items() if _classify_stage(model, key) in stage_names)


def _print_summary_table(
    *,
    title: str,
    result_dict: dict,
    keys: list[str],
    database_mode: common.DatabaseMode,
) -> None:
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"  {'metric':<34s}  {database_mode.name:>14s}")
    print(f"  {'-' * 104}")

    available_keys = [key for key in keys if key in result_dict]
    for key in available_keys:
        print(f"  {key:<34s}  {_fmt_value(result_dict.get(key, ''))}")


def _print_kernel_breakdown(
    *,
    title: str,
    model: BaseModel,
    summary: InferenceSummary,
    phase: str,
    database_mode: common.DatabaseMode,
) -> dict[str, float]:
    latencies = _phase_dict(summary, phase)
    total = sum(float(value) for value in latencies.values())
    a2f_total = _stage_total(model, latencies, {"a2f"})
    f2a_total = _stage_total(model, latencies, {"f2a"})
    transfer_total = a2f_total + f2a_total
    attn_total = _stage_total(model, latencies, {"attn"})
    ffn_total = _stage_total(model, latencies, {"ffn"})
    attn_pct = 100.0 * attn_total / total if total > 0 else 0.0
    ffn_pct = 100.0 * ffn_total / total if total > 0 else 0.0

    print(f"\n{'=' * 120}")
    print(f"  {title}  ({database_mode.name}={total:.4f} ms)")
    if total > 0:
        print(
            f"  attn={attn_total:.4f}ms ({attn_pct:.1f}%)  "
            f"ffn={ffn_total:.4f}ms ({ffn_pct:.1f}%)  "
            f"afd_transfer={transfer_total:.4f}ms ({100.0 * transfer_total / total:.1f}%)"
        )

    print(f"  {'op':<44s} {'stage':>5s}  {database_mode.name:>10s}  {'share':>8s}")
    print(f"  {'-' * 118}")

    for op, latency in latencies.items():
        stage = _classify_stage(model, op)
        pct = 100.0 * float(latency) / total if total > 0 else 0.0
        print(f"  {op:<44s} {stage:>5s}  {float(latency):10.4f}  {pct:7.2f}%")

    print(f"  {'-' * 118}")
    for label, value in [
        ("AFD A2F TOTAL", a2f_total),
        ("AFD F2A TOTAL", f2a_total),
        ("AFD TRANSFER TOTAL", transfer_total),
        ("TOTAL", total),
    ]:
        pct = 100.0 * value / total if total > 0 else 0.0
        print(f"  {label:<44s} {'':>5s}  {value:10.4f}  {pct:7.2f}%")

    return {
        "total_ms": total,
        "attn_ms": attn_total,
        "ffn_ms": ffn_total,
        "attn_pct": attn_pct,
        "ffn_pct": ffn_pct,
        "a2f_ms": a2f_total,
        "f2a_ms": f2a_total,
        "afd_transfer_ms": transfer_total,
    }


def _run_agg_modes(
    *,
    args: argparse.Namespace,
    runtime_config: config.RuntimeConfig,
) -> AggRunResult:
    database = _build_database(
        system=args.system,
        backend=args.backend,
        version=args.version,
        use_astrasim=args.use_astrasim,
    )
    backend = get_backend(args.backend)
    model_config = _build_model_config(args)

    database.set_default_database_mode(_database_mode(args))
    if hasattr(backend, "_agg_cache"):
        backend._agg_cache.clear()
    model = get_model(args.model, model_config, backend.name.value)
    session = InferenceSession(model=model, database=database, backend=backend)
    agg_summary = session.run_agg(
        runtime_config,
        ctx_tokens=_ctx_tokens_for_case(args, runtime_config),
    )
    context_kernels = session.run_static(runtime_config=runtime_config, mode="static_ctx")
    generation_kernels = session.run_static(runtime_config=runtime_config, mode="static_gen")
    return AggRunResult(
        summary=agg_summary,
        context_kernels=context_kernels,
        generation_kernels=generation_kernels,
    )


def _run_disagg_afd_modes(
    *,
    args: argparse.Namespace,
    runtime_config: config.RuntimeConfig,
) -> dict:
    prefill_database = _build_database(
        system=args.system,
        backend=args.backend,
        version=args.version,
        use_astrasim=args.use_astrasim,
    )
    decode_system = args.decode_system or args.system
    decode_backend_name = args.decode_backend or args.backend
    decode_version = args.decode_version or args.version
    if decode_system == args.system and decode_backend_name == args.backend and decode_version == args.version:
        decode_database = prefill_database
    else:
        decode_database = _build_database(
            system=decode_system,
            backend=decode_backend_name,
            version=decode_version,
            use_astrasim=args.use_astrasim,
        )

    prefill_backend = get_backend(args.backend)
    decode_backend = prefill_backend if decode_backend_name == args.backend else get_backend(decode_backend_name)
    model_config = _build_model_config(args)

    session = DisaggInferenceSession(
        prefill_database=prefill_database,
        prefill_backend=prefill_backend,
        decode_database=decode_database,
        decode_backend=decode_backend,
        gpu_layout_strategy=args.gpu_layout_strategy,
    )

    database_mode = _database_mode(args)
    prefill_database.set_default_database_mode(database_mode)
    decode_database.set_default_database_mode(database_mode)
    return session.run_disagg_afd(
        model_path=args.model,
        runtime_config=runtime_config,
        prefill_model_config=model_config,
        prefill_batch_size=args.prefill_batch_size,
        prefill_num_worker=args.prefill_workers,
        decode_model_config=model_config,
        decode_batch_size=args.decode_batch_size,
        decode_num_worker=args.decode_workers,
    )


def _append_tracking_rows(
    rows: list[dict],
    *,
    serving: str,
    args: argparse.Namespace,
    isl: int,
    osl: int,
    phase: str,
    stats: dict[str, float],
) -> None:
    total_ms = stats["total_ms"]
    afd_ms = stats["afd_transfer_ms"]
    rows.append(
        {
            "serving": serving,
            "afd_layout": _afd_label(args),
            "tp": args.tp,
            "attention_dp": args.attention_dp,
            "moe_tp": args.moe_tp,
            "moe_ep": args.moe_ep,
            "num_attn_gpus": _num_attn_gpus(args),
            "num_ffn_gpus": _num_ffn_gpus(args),
            "isl": isl,
            "osl": osl,
            "phase": phase,
            "db_mode": _database_mode(args).name,
            "total_ms": total_ms,
            "attn_ms": stats["attn_ms"],
            "ffn_ms": stats["ffn_ms"],
            "attn_pct": stats["attn_pct"],
            "ffn_pct": stats["ffn_pct"],
            "a2f_ms": stats["a2f_ms"],
            "f2a_ms": stats["f2a_ms"],
            "afd_transfer_ms": afd_ms,
            "afd_transfer_pct": (100.0 * afd_ms / total_ms) if total_ms > 0 else 0.0,
        }
    )


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} tracking rows to {output_path}")


def _print_case_header(
    *,
    serving: str,
    runtime_config: config.RuntimeConfig,
    args: argparse.Namespace,
) -> None:
    print(f"\n{'#' * 120}")
    print(
        f"  AFD {serving.upper()} TRACKING  layout={_afd_label(args)}  "
        f"isl={runtime_config.isl}  osl={runtime_config.osl}  "
        f"batch_size={runtime_config.batch_size}"
    )
    print(
        f"  model={args.model}  system={args.system}  backend={args.backend}  "
        f"db_mode={_database_mode(args).name}  use_astrasim={args.use_astrasim}"
    )
    print(
        f"  model_config: tp={args.tp} pp={args.pp} "
        f"attention_dp={args.attention_dp} "
        f"moe_tp={args.moe_tp} moe_ep={args.moe_ep} "
        f"afd_microbatches={args.afd_num_microbatches}"
    )
    print(
        f"  derived_gpus: num_attn_gpus={_num_attn_gpus(args)} "
        f"num_ffn_gpus={_num_ffn_gpus(args)} "
        f"total_worker_gpus={_num_attn_gpus(args) + _num_ffn_gpus(args)}"
    )
    if serving == "agg":
        print(f"  agg ctx_tokens={_ctx_tokens_for_case(args, runtime_config)}")


def run(args: argparse.Namespace) -> None:
    _require_native_astrasim(args.use_astrasim)
    database_mode = _database_mode(args)
    isls = _parse_int_list(args.isls)
    osls = _parse_int_list(args.osls)
    tracking_rows: list[dict] = []

    servings = ["agg", "disagg"] if args.serving == "both" else [args.serving]
    model_config = _build_model_config(args)
    prefill_backend = get_backend(args.backend)
    prefill_model = get_model(args.model, model_config, prefill_backend.name.value)
    decode_backend_name = args.decode_backend or args.backend
    if decode_backend_name == args.backend:
        decode_model = prefill_model
    else:
        decode_backend = get_backend(decode_backend_name)
        decode_model = get_model(args.model, model_config, decode_backend.name.value)

    for isl in isls:
        for osl in osls:
            runtime_config = _runtime_config(args, isl, osl)

            if "agg" in servings:
                _print_case_header(
                    serving="agg",
                    runtime_config=runtime_config,
                    args=args,
                )
                agg_result = _run_agg_modes(
                    args=args,
                    runtime_config=runtime_config,
                )
                _print_summary_table(
                    title="AGG SUMMARY",
                    result_dict=agg_result.summary.get_result_dict() or {},
                    keys=SUMMARY_KEYS_AGG,
                    database_mode=database_mode,
                )
                agg_phase_specs = [
                    ("context", "AGG STATIC_CTX", agg_result.context_kernels),
                    ("generation", "AGG STATIC_GEN", agg_result.generation_kernels),
                ]
                for phase, label, summary in agg_phase_specs:
                    stats = _print_kernel_breakdown(
                        title=f"{label} KERNEL BREAKDOWN",
                        model=prefill_model,
                        summary=summary,
                        phase=phase,
                        database_mode=database_mode,
                    )
                    _append_tracking_rows(
                        tracking_rows,
                        serving="agg",
                        args=args,
                        isl=isl,
                        osl=osl,
                        phase=phase,
                        stats=stats,
                    )

            if "disagg" in servings:
                _print_case_header(
                    serving="disagg",
                    runtime_config=runtime_config,
                    args=args,
                )
                results = _run_disagg_afd_modes(
                    args=args,
                    runtime_config=runtime_config,
                )
                _print_summary_table(
                    title="DISAGG SUMMARY",
                    result_dict=results["disagg_summary"].get_result_dict() or {},
                    keys=SUMMARY_KEYS_DISAGG,
                    database_mode=database_mode,
                )

                phase_specs = [
                    ("prefill_attn", "PREFILL ATTN", "context", prefill_model),
                    ("prefill_ffn", "PREFILL FFN", "context", prefill_model),
                    ("decode_attn", "DECODE ATTN", "generation", decode_model),
                    ("decode_ffn", "DECODE FFN", "generation", decode_model),
                ]
                for result_key, label, phase, model in phase_specs:
                    stats = _print_kernel_breakdown(
                        title=label,
                        model=model,
                        summary=results[result_key],
                        phase=phase,
                        database_mode=database_mode,
                    )
                    _append_tracking_rows(
                        tracking_rows,
                        serving="disagg",
                        args=args,
                        isl=isl,
                        osl=osl,
                        phase=label.lower().replace(" ", "_"),
                        stats=stats,
                    )

    if args.csv:
        _write_csv(args.csv, tracking_rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track AFD dispatch/transfer kernels over ISL/OSL cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--serving", choices=["agg", "disagg", "both"], default="agg")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--decode-system", default=None)
    parser.add_argument("--decode-backend", default=None)
    parser.add_argument("--decode-version", default=None)
    parser.add_argument(
        "--database-mode",
        choices=[mode.name for mode in common.DatabaseMode],
        default=DEFAULT_DATABASE_MODE.name,
        help="Performance-database source for attention and FFN compute.",
    )

    parser.add_argument("--isls", default="6400", help="Comma-separated ISL values.")
    parser.add_argument("--osls", default="1000", help="Comma-separated OSL values.")
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--ctx-tokens", type=int, default=None, help="Agg ctx token budget. Default: per-case ISL.")

    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument(
        "--dp",
        "--attention-dp",
        dest="attention_dp",
        type=int,
        default=4,
        help="Attention DP. num_attn_gpus is derived as tp * dp.",
    )
    parser.add_argument("--moe-tp", type=int, default=1)
    parser.add_argument(
        "--moe-ep",
        type=int,
        default=4,
        help="MoE EP. num_ffn_gpus is derived as moe_tp * moe_ep.",
    )
    parser.add_argument("--afd-num-microbatches", type=int, default=1)
    parser.add_argument("--enable-wideep", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--prefill-batch-size", type=int, default=2)
    parser.add_argument("--decode-batch-size", type=int, default=2)
    parser.add_argument("--prefill-workers", type=int, default=1)
    parser.add_argument("--decode-workers", type=int, default=1)
    parser.add_argument(
        "--gpu-layout-strategy",
        default="segregated_by_phase",
        choices=["segregated_by_phase", "paired_prefill_decode_per_node"],
    )

    parser.add_argument(
        "--use-astrasim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route AFD P2P queries in PerfDatabase through AstraSim when available.",
    )
    parser.add_argument("--csv", default=None, help="Optional compact tracking CSV output.")
    return parser


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
