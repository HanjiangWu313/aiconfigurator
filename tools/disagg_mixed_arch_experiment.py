#!/usr/bin/env python3
"""
Run disaggregated inference with separate prefill and decode system specs.

This is intended for analytical mixed-architecture studies, for example:
  - keep decode on a fixed GPU system
  - compare prefill on rubin_cpx_proxy vs r200_proxy

The proxy systems added for Rubin / R200 reuse a real node-network template
and backend data layout, but override the local chip compute and memory knobs.
Because the SDK does not yet expose an explicit FP4 architectural field, the
published dense-compute number is mapped onto the generic dense-compute fields
used by the current analytical model.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiconfigurator.sdk import common, config as sdk_config, perf_database
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import DisaggInferenceSession


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


def _parse_enum(enum_cls, name: str):
    return enum_cls[name]


def _make_database(
    system: str,
    backend: str,
    version: str,
    systems_dir: str,
    database_mode: common.DatabaseMode,
) -> tuple[perf_database.PerfDatabase, str]:
    resolved_version = _resolve_version(system, backend, version)
    db = perf_database.PerfDatabase(system, backend, resolved_version, systems_dir=systems_dir)
    db._default_database_mode = database_mode
    return db, resolved_version


def _build_model_config(args, prefix: str) -> sdk_config.ModelConfig:
    return sdk_config.ModelConfig(
        tp_size=getattr(args, f"{prefix}_tp"),
        pp_size=getattr(args, f"{prefix}_pp"),
        attention_dp_size=getattr(args, f"{prefix}_dp"),
        moe_tp_size=getattr(args, f"{prefix}_moe_tp"),
        moe_ep_size=getattr(args, f"{prefix}_moe_ep"),
        gemm_quant_mode=_parse_enum(common.GEMMQuantMode, args.gemm_quant),
        moe_quant_mode=_parse_enum(common.MoEQuantMode, args.moe_quant),
        kvcache_quant_mode=_parse_enum(common.KVCacheQuantMode, args.kvcache_quant),
        fmha_quant_mode=_parse_enum(common.FMHAQuantMode, args.fmha_quant),
        comm_quant_mode=_parse_enum(common.CommQuantMode, args.comm_quant),
        workload_distribution=args.workload_distribution,
        overwrite_num_layers=args.overwrite_num_layers,
    )


def _build_runtime_config(args) -> sdk_config.RuntimeConfig:
    return sdk_config.RuntimeConfig(
        batch_size=1,
        beam_width=args.beam_width,
        isl=args.isl,
        osl=args.osl,
        prefix=args.prefix,
    )


def _collect_summary_row(
    prefill_system: str,
    prefill_version: str,
    prefill_db: perf_database.PerfDatabase,
    decode_system: str,
    decode_version: str,
    decode_db: perf_database.PerfDatabase,
    args,
) -> dict[str, object]:
    prefill_backend = get_backend(args.backend)
    decode_backend = get_backend(args.backend)
    session = DisaggInferenceSession(
        prefill_database=prefill_db,
        prefill_backend=prefill_backend,
        decode_database=decode_db,
        decode_backend=decode_backend,
        network_file=args.network_file,
    )
    session.set_latency_correction_scales(
        args.prefill_latency_correction,
        args.decode_latency_correction,
    )

    runtime_config = _build_runtime_config(args)
    prefill_model_config = _build_model_config(args, "prefill")
    decode_model_config = _build_model_config(args, "decode")

    summary = session.run_disagg(
        model_path=args.model_path,
        runtime_config=runtime_config,
        prefill_model_config=prefill_model_config,
        prefill_batch_size=args.prefill_batch_size,
        prefill_num_worker=args.prefill_num_worker,
        decode_model_config=decode_model_config,
        decode_batch_size=args.decode_batch_size,
        decode_num_worker=args.decode_num_worker,
    )

    row = summary.get_result_dict() or {}
    kv_network_latency_ms = float(row.get("kv_network_latency_ms", 0.0))
    ttft_ms = float(row.get("ttft", 0.0))
    request_latency_ms = float(row.get("request_latency", 0.0))

    row.update(
        {
            "status": "ok",
            "prefill_system": prefill_system,
            "prefill_version": prefill_version,
            "decode_system": decode_system,
            "decode_version": decode_version,
            "backend": args.backend,
            "database_mode": args.database_mode,
            "model_path": args.model_path,
            "prefill_float16_tc_flops": prefill_db.system_spec["gpu"]["float16_tc_flops"],
            "prefill_mem_bw": prefill_db.system_spec["gpu"]["mem_bw"],
            "prefill_mem_capacity": prefill_db.system_spec["gpu"]["mem_capacity"],
            "decode_float16_tc_flops": decode_db.system_spec["gpu"]["float16_tc_flops"],
            "decode_mem_bw": decode_db.system_spec["gpu"]["mem_bw"],
            "decode_mem_capacity": decode_db.system_spec["gpu"]["mem_capacity"],
            "ttft_plus_kv_ms": ttft_ms + kv_network_latency_ms,
            "request_latency_plus_kv_ms": request_latency_ms + kv_network_latency_ms,
            "kv_pct_of_ttft": summary.get_kv_cache_transfer_pct(),
            "oom": bool(summary.check_oom()),
        }
    )
    return row


def _error_row(prefill_system: str, decode_system: str, args, exc: Exception) -> dict[str, object]:
    return {
        "status": "error",
        "prefill_system": prefill_system,
        "decode_system": decode_system,
        "backend": args.backend,
        "database_mode": args.database_mode,
        "model_path": args.model_path,
        "error": str(exc),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefill-systems",
        nargs="+",
        required=True,
        help="One or more system YAML names for the prefill side, e.g. rubin_cpx_proxy r200_proxy",
    )
    parser.add_argument("--decode-system", required=True, help="Fixed system YAML name for decode")
    parser.add_argument("--backend", required=True, choices=[b.value for b in common.BackendName])
    parser.add_argument("--prefill-version", default="latest", help="Prefill database version or 'latest'")
    parser.add_argument("--decode-version", default="latest", help="Decode database version or 'latest'")
    parser.add_argument("--systems-dir", default=None, help="Override systems directory")
    parser.add_argument(
        "--database-mode",
        default="SOL",
        choices=[mode.name for mode in common.DatabaseMode],
        help="Use SOL for analytical proxy studies; SILICON is not recommended for the proxy systems.",
    )
    parser.add_argument("--network-file", default=None, help="Optional AstraSim topology YAML")
    parser.add_argument("--output-csv", default=None, help="Optional path to save CSV output")

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--osl", type=int, required=True)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=1)

    parser.add_argument("--prefill-batch-size", type=int, required=True)
    parser.add_argument("--prefill-num-worker", type=int, required=True)
    parser.add_argument("--decode-batch-size", type=int, required=True)
    parser.add_argument("--decode-num-worker", type=int, required=True)

    parser.add_argument("--prefill-tp", type=int, default=1)
    parser.add_argument("--prefill-pp", type=int, default=1)
    parser.add_argument("--prefill-dp", type=int, default=1)
    parser.add_argument("--prefill-moe-tp", type=int, default=1)
    parser.add_argument("--prefill-moe-ep", type=int, default=1)
    parser.add_argument("--decode-tp", type=int, default=1)
    parser.add_argument("--decode-pp", type=int, default=1)
    parser.add_argument("--decode-dp", type=int, default=1)
    parser.add_argument("--decode-moe-tp", type=int, default=1)
    parser.add_argument("--decode-moe-ep", type=int, default=1)

    parser.add_argument("--overwrite-num-layers", type=int, default=0)
    parser.add_argument("--gemm-quant", choices=[q.name for q in common.GEMMQuantMode], default="float16")
    parser.add_argument("--moe-quant", choices=[q.name for q in common.MoEQuantMode], default="float16")
    parser.add_argument("--fmha-quant", choices=[q.name for q in common.FMHAQuantMode], default="float16")
    parser.add_argument("--kvcache-quant", choices=[q.name for q in common.KVCacheQuantMode], default="float16")
    parser.add_argument("--comm-quant", choices=[q.name for q in common.CommQuantMode], default="half")
    parser.add_argument("--workload-distribution", default="power_law")

    parser.add_argument("--prefill-latency-correction", type=float, default=1.0)
    parser.add_argument("--decode-latency-correction", type=float, default=1.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    systems_dir = _systems_dir_arg(args.systems_dir)
    database_mode = common.DatabaseMode[args.database_mode]

    rows: list[dict[str, object]] = []
    for prefill_system in args.prefill_systems:
        try:
            prefill_db, prefill_version = _make_database(
                prefill_system,
                args.backend,
                args.prefill_version,
                systems_dir,
                database_mode,
            )
            decode_db, decode_version = _make_database(
                args.decode_system,
                args.backend,
                args.decode_version,
                systems_dir,
                database_mode,
            )
            row = _collect_summary_row(
                prefill_system,
                prefill_version,
                prefill_db,
                args.decode_system,
                decode_version,
                decode_db,
                args,
            )
        except Exception as exc:
            row = _error_row(prefill_system, args.decode_system, args, exc)
        rows.append(row)

    if args.output_csv and rows:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {output_path}")

    if rows:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
