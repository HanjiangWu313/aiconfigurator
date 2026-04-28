#!/usr/bin/env python3
"""
One-parameter-at-a-time system sensitivity sweeps for analytical operators or static backend runs.

Examples
--------
Operator-level analytical sweep with math-vs-mem breakdown:

    python tools/system_param_sweep.py operator \
      --system h100_sxm \
      --backend trtllm \
      --param gpu.mem_bw \
      --scale-factors 0.5 0.75 1.0 1.25 1.5 \
      --operator gemm \
      --m 16384 --n 16384 --k 16384 \
      --database-mode SOL_FULL

Static backend sensitivity sweep:

    python tools/system_param_sweep.py static \
      --system h100_sxm \
      --backend trtllm \
      --param gpu.mem_bw \
      --scale-factors 0.5 0.75 1.0 1.25 1.5 \
      --model-path meta-llama/Meta-Llama-3.1-8B \
      --run-mode static_ctx \
      --batch-size 8 --isl 4096 --osl 1 \
      --database-mode SOL
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiconfigurator.sdk import common, config as sdk_config, perf_database


def _systems_dir_arg(value: str | None) -> str:
    if value:
        return value
    return str(perf_database.get_system_config_path())


def _resolve_version(system: str, backend: str, version: str) -> str:
    if version != "latest":
        return version
    resolved = perf_database.get_latest_database_version(system, backend)
    if resolved is None:
        raise ValueError(f"Could not resolve latest database version for system={system}, backend={backend}")
    return resolved


def _fresh_database(system: str, backend: str, version: str, systems_dir: str) -> perf_database.PerfDatabase:
    resolved_version = _resolve_version(system, backend, version)
    return perf_database.PerfDatabase(system, backend, resolved_version, systems_dir=systems_dir)


def _get_nested(mapping: dict, path: str):
    current = mapping
    parts = path.split(".")
    for key in parts[:-1]:
        current = current[key]
    return current, parts[-1]


def _cast_like(value: float, baseline):
    if isinstance(baseline, int):
        return int(round(value))
    return float(value)


def _parse_sweep_values(baseline, values: list[float] | None, scale_factors: list[float] | None) -> list[tuple[float, float | None]]:
    if values:
        return [(v, None) for v in values]
    if scale_factors:
        return [(_cast_like(baseline * factor, baseline), factor) for factor in scale_factors]
    return [(baseline, 1.0)]


def _mutate_system_spec(db: perf_database.PerfDatabase, param_path: str, new_value) -> None:
    mutated = copy.deepcopy(db.system_spec)
    parent, leaf = _get_nested(mutated, param_path)
    parent[leaf] = new_value
    db.system_spec = mutated
    if getattr(db, "_astrasim", None) is not None:
        db._astrasim._system_spec = mutated


def _result_to_dict(result) -> dict[str, float]:
    if isinstance(result, tuple):
        latency_ms, math_ms, mem_ms = result
        bounded_by = "math" if math_ms >= mem_ms else "mem"
        return {
            "latency_ms": float(latency_ms),
            "math_ms": float(math_ms),
            "mem_ms": float(mem_ms),
            "bound": bounded_by,
        }
    return {"latency_ms": float(result)}


def _parse_enum(enum_cls, name: str):
    return enum_cls[name]


def _evaluate_operator(db: perf_database.PerfDatabase, args, database_mode: common.DatabaseMode) -> dict[str, float]:
    op = args.operator
    if op == "gemm":
        result = db.query_gemm(
            args.m,
            args.n,
            args.k,
            _parse_enum(common.GEMMQuantMode, args.gemm_quant),
            database_mode=database_mode,
        )
    elif op == "context_attention":
        result = db.query_context_attention(
            b=args.batch_size,
            s=args.seq_len,
            prefix=args.prefix,
            n=args.num_heads,
            n_kv=args.num_kv_heads,
            kvcache_quant_mode=_parse_enum(common.KVCacheQuantMode, args.kvcache_quant),
            fmha_quant_mode=_parse_enum(common.FMHAQuantMode, args.fmha_quant),
            database_mode=database_mode,
            window_size=args.window_size,
            head_size=args.head_size,
        )
    elif op == "generation_attention":
        result = db.query_generation_attention(
            b=args.batch_size,
            s=args.seq_len,
            n=args.num_heads,
            n_kv=args.num_kv_heads,
            kvcache_quant_mode=_parse_enum(common.KVCacheQuantMode, args.kvcache_quant),
            database_mode=database_mode,
            window_size=args.window_size,
            head_size=args.head_size,
        )
    elif op == "context_mla":
        result = db.query_context_mla(
            b=args.batch_size,
            s=args.seq_len,
            prefix=args.prefix,
            num_heads=args.num_heads,
            kvcache_quant_mode=_parse_enum(common.KVCacheQuantMode, args.kvcache_quant),
            fmha_quant_mode=_parse_enum(common.FMHAQuantMode, args.fmha_quant),
            database_mode=database_mode,
        )
    elif op == "generation_mla":
        result = db.query_generation_mla(
            b=args.batch_size,
            s=args.seq_len,
            num_heads=args.num_heads,
            kvcache_quant_mode=_parse_enum(common.KVCacheQuantMode, args.kvcache_quant),
            database_mode=database_mode,
        )
    elif op == "moe":
        result = db.query_moe(
            num_tokens=args.num_tokens,
            hidden_size=args.hidden_size,
            inter_size=args.inter_size,
            topk=args.topk,
            num_experts=args.num_experts,
            moe_tp_size=args.moe_tp,
            moe_ep_size=args.moe_ep,
            quant_mode=_parse_enum(common.MoEQuantMode, args.moe_quant),
            workload_distribution=args.workload_distribution,
            is_context=(args.phase == "context"),
            moe_backend=args.moe_backend,
            database_mode=database_mode,
        )
    elif op == "nccl":
        result = db.query_nccl(
            dtype=_parse_enum(common.CommQuantMode, args.comm_quant),
            num_gpus=args.num_gpus,
            operation=args.collective_op,
            message_size=args.message_size,
            database_mode=database_mode,
        )
    elif op == "p2p":
        result = db.query_p2p(
            message_bytes=args.message_bytes,
            database_mode=database_mode,
        )
    elif op == "afd_p2p":
        result = db.query_afd_p2p(
            sender_bytes=args.sender_bytes,
            receiver_bytes=args.receiver_bytes,
            num_gpus=args.num_gpus,
            database_mode=database_mode,
        )
    elif op == "mem":
        result = db.query_mem_op(
            mem_bytes=args.mem_bytes,
            database_mode=database_mode,
        )
    else:
        raise ValueError(f"Unsupported operator: {op}")
    return _result_to_dict(result)


def _build_model_config(args) -> sdk_config.ModelConfig:
    return sdk_config.ModelConfig(
        tp_size=args.tp,
        pp_size=args.pp,
        attention_dp_size=args.dp,
        moe_tp_size=args.moe_tp,
        moe_ep_size=args.moe_ep,
        gemm_quant_mode=_parse_enum(common.GEMMQuantMode, args.gemm_quant),
        moe_quant_mode=_parse_enum(common.MoEQuantMode, args.moe_quant),
        kvcache_quant_mode=_parse_enum(common.KVCacheQuantMode, args.kvcache_quant),
        fmha_quant_mode=_parse_enum(common.FMHAQuantMode, args.fmha_quant),
        comm_quant_mode=_parse_enum(common.CommQuantMode, args.comm_quant),
        workload_distribution=args.workload_distribution,
        overwrite_num_layers=args.overwrite_num_layers,
    )


def _evaluate_static(db: perf_database.PerfDatabase, args, database_mode: common.DatabaseMode) -> dict:
    from aiconfigurator.sdk import models
    from aiconfigurator.sdk.backends.factory import get_backend

    if database_mode == common.DatabaseMode.SOL_FULL:
        raise ValueError("static mode does not support SOL_FULL because run_static expects scalar query results")

    backend = get_backend(args.backend)
    model_config = _build_model_config(args)
    runtime_config = sdk_config.RuntimeConfig(
        batch_size=args.batch_size,
        beam_width=args.beam_width,
        isl=args.isl,
        osl=args.osl,
        prefix=args.prefix,
    )

    db._default_database_mode = database_mode
    model = models.get_model(args.model_path, model_config, backend.name.value)
    summary = backend.run_static(model, db, runtime_config, args.run_mode)
    row = summary.get_result_dict() or {}
    row["oom"] = bool(summary.check_oom())
    return row


def _add_common_sweep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--system", required=True, help="System YAML name without .yaml, e.g. h100_sxm")
    parser.add_argument("--backend", required=True, choices=[b.value for b in common.BackendName])
    parser.add_argument("--version", default="latest", help="Database version or 'latest'")
    parser.add_argument("--systems-dir", default=None, help="Override systems directory")
    parser.add_argument("--param", required=True, help="Nested system parameter path, e.g. gpu.mem_bw")
    parser.add_argument("--values", nargs="+", type=float, help="Explicit sweep values")
    parser.add_argument("--scale-factors", nargs="+", type=float, help="Scale factors applied to the baseline value")
    parser.add_argument(
        "--database-mode",
        default="SOL_FULL",
        choices=[mode.name for mode in common.DatabaseMode],
        help="Analytical/silicon mode. For static mode, prefer SOL/EMPIRICAL/HYBRID/SILICON.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional path to save CSV output")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    operator = subparsers.add_parser("operator", help="Sweep a system parameter for a single PerfDatabase query")
    _add_common_sweep_args(operator)
    operator.add_argument(
        "--operator",
        required=True,
        choices=[
            "gemm",
            "context_attention",
            "generation_attention",
            "context_mla",
            "generation_mla",
            "moe",
            "nccl",
            "p2p",
            "afd_p2p",
            "mem",
        ],
    )
    operator.add_argument("--m", type=int, default=4096)
    operator.add_argument("--n", type=int, default=4096)
    operator.add_argument("--k", type=int, default=4096)
    operator.add_argument("--batch-size", type=int, default=1)
    operator.add_argument("--seq-len", type=int, default=4096)
    operator.add_argument("--prefix", type=int, default=0)
    operator.add_argument("--num-heads", type=int, default=32)
    operator.add_argument("--num-kv-heads", type=int, default=8)
    operator.add_argument("--head-size", type=int, default=128)
    operator.add_argument("--window-size", type=int, default=0)
    operator.add_argument("--num-tokens", type=int, default=4096)
    operator.add_argument("--hidden-size", type=int, default=8192)
    operator.add_argument("--inter-size", type=int, default=28672)
    operator.add_argument("--topk", type=int, default=2)
    operator.add_argument("--num-experts", type=int, default=8)
    operator.add_argument("--moe-tp", type=int, default=1)
    operator.add_argument("--moe-ep", type=int, default=1)
    operator.add_argument("--phase", choices=["context", "generation"], default="context")
    operator.add_argument("--moe-backend", default=None)
    operator.add_argument("--num-gpus", type=int, default=8)
    operator.add_argument("--collective-op", choices=["all_reduce", "all_gather", "reduce_scatter", "alltoall"], default="all_reduce")
    operator.add_argument("--message-size", type=int, default=1 << 20, help="Collective payload in elements")
    operator.add_argument("--message-bytes", type=int, default=1 << 30)
    operator.add_argument("--sender-bytes", type=int, default=1 << 30)
    operator.add_argument("--receiver-bytes", type=int, default=1 << 30)
    operator.add_argument("--mem-bytes", type=int, default=1 << 30)
    operator.add_argument("--gemm-quant", choices=[q.name for q in common.GEMMQuantMode], default="float16")
    operator.add_argument("--moe-quant", choices=[q.name for q in common.MoEQuantMode], default="float16")
    operator.add_argument("--fmha-quant", choices=[q.name for q in common.FMHAQuantMode], default="float16")
    operator.add_argument("--kvcache-quant", choices=[q.name for q in common.KVCacheQuantMode], default="float16")
    operator.add_argument("--comm-quant", choices=[q.name for q in common.CommQuantMode], default="half")
    operator.add_argument("--workload-distribution", default="power_law")

    static = subparsers.add_parser("static", help="Sweep a system parameter for a backend static run")
    _add_common_sweep_args(static)
    static.add_argument("--model-path", required=True)
    static.add_argument("--run-mode", choices=["static", "static_ctx", "static_gen"], default="static")
    static.add_argument("--batch-size", type=int, required=True)
    static.add_argument("--isl", type=int, required=True)
    static.add_argument("--osl", type=int, required=True)
    static.add_argument("--beam-width", type=int, default=1)
    static.add_argument("--prefix", type=int, default=0)
    static.add_argument("--tp", type=int, default=1)
    static.add_argument("--pp", type=int, default=1)
    static.add_argument("--dp", type=int, default=1)
    static.add_argument("--moe-tp", type=int, default=1)
    static.add_argument("--moe-ep", type=int, default=1)
    static.add_argument("--overwrite-num-layers", type=int, default=0)
    static.add_argument("--gemm-quant", choices=[q.name for q in common.GEMMQuantMode], default="float16")
    static.add_argument("--moe-quant", choices=[q.name for q in common.MoEQuantMode], default="float16")
    static.add_argument("--fmha-quant", choices=[q.name for q in common.FMHAQuantMode], default="float16")
    static.add_argument("--kvcache-quant", choices=[q.name for q in common.KVCacheQuantMode], default="float16")
    static.add_argument("--comm-quant", choices=[q.name for q in common.CommQuantMode], default="half")
    static.add_argument("--workload-distribution", default="power_law")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    systems_dir = _systems_dir_arg(args.systems_dir)
    baseline_db = _fresh_database(args.system, args.backend, args.version, systems_dir)
    parent, leaf = _get_nested(baseline_db.system_spec, args.param)
    baseline_value = parent[leaf]
    sweep_values = _parse_sweep_values(baseline_value, args.values, args.scale_factors)
    database_mode = common.DatabaseMode[args.database_mode]

    rows = []
    for sweep_value, scale_factor in sweep_values:
        db = _fresh_database(args.system, args.backend, args.version, systems_dir)
        _mutate_system_spec(db, args.param, sweep_value)

        if args.command == "operator":
            metrics = _evaluate_operator(db, args, database_mode)
        else:
            metrics = _evaluate_static(db, args, database_mode)

        rows.append(
            {
                "system": args.system,
                "backend": args.backend,
                "database_mode": args.database_mode,
                "param": args.param,
                "baseline_value": baseline_value,
                "sweep_value": sweep_value,
                "scale_factor": scale_factor,
                **metrics,
            }
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {output_path}")

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
