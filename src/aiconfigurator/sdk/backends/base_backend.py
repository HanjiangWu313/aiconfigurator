# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

if TYPE_CHECKING:
    from aiconfigurator.sdk.config import AfdConfig

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """
    Base class for all backends.
    All backends should inherit from this class and implement the abstract methods.
    All backends should implement the following methods:

    Attributes:

    Methods:
        run_static: this is common for all backends. It's implemented in this class.
            If there might be some backend-specific logic, it should be implemented in the subclass.
        run_agg: this is backend-specific. It should be implemented in the subclass.
        find_best_agg_result_under_constraints: this is backend-specific.
            It should be implemented in the subclass.
        _get_memory_usage: this is backend-specific. It should be implemented in the subclass.
    """

    def _run_context_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        isl: int,
        prefix: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        context_latency_dict = defaultdict(float)
        context_energy_wms_dict = defaultdict(float)

        effective_isl = isl - prefix
        if effective_isl <= 0:
            raise ValueError(f"isl must be greater than 0 after removing prefix, but got {effective_isl}")

        for op in model.context_ops:
            x = batch_size * effective_isl if "logits_gemm" not in op._name else batch_size
            result = op.query(
                database,
                x=x,
                batch_size=batch_size,
                beam_width=1,
                s=effective_isl,
                prefix=prefix,
                model_name=getattr(model, "model_name", ""),
                seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
            )
            context_latency_dict[op._name] += float(result)
            context_energy_wms_dict[op._name] += getattr(result, "energy", 0.0)

        return context_latency_dict, context_energy_wms_dict

    def _run_generation_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        stride: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        generation_latency_dict = defaultdict(float)
        generation_energy_wms_dict = defaultdict(float)

        batch_size = batch_size * (model._nextn + 1)

        for i in range(0, osl - 1, stride):
            latency_dict = defaultdict(float)
            energy_wms_dict = defaultdict(float)

            for op in model.generation_ops:
                result = op.query(
                    database,
                    x=batch_size * beam_width,
                    batch_size=batch_size,
                    beam_width=beam_width,
                    s=isl + i + 1,
                    model_name=getattr(model, "model_name", ""),
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                )
                latency_dict[op._name] += float(result)
                energy_wms_dict[op._name] += getattr(result, "energy", 0.0)

            repeat_count = min(stride, osl - 1 - i)
            for op in latency_dict:
                generation_latency_dict[op] += latency_dict[op] * repeat_count
                generation_energy_wms_dict[op] += energy_wms_dict[op] * repeat_count

        return generation_latency_dict, generation_energy_wms_dict

    def _run_static_breakdown(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        batch_size, beam_width, isl, osl, prefix = (
            runtime_config.batch_size,
            runtime_config.beam_width,
            runtime_config.isl,
            runtime_config.osl,
            runtime_config.prefix,
        )

        context_latency_dict, context_energy_wms_dict = {}, {}
        generation_latency_dict, generation_energy_wms_dict = {}, {}

        if mode == "static_ctx":
            context_latency_dict, context_energy_wms_dict = self._run_context_phase(
                model, database, runtime_config, batch_size, isl, prefix
            )
        elif mode == "static_gen":
            generation_latency_dict, generation_energy_wms_dict = self._run_generation_phase(
                model, database, runtime_config, batch_size, beam_width, isl, osl, stride
            )
        else:
            context_latency_dict, context_energy_wms_dict = self._run_context_phase(
                model, database, runtime_config, batch_size, isl, prefix
            )
            generation_latency_dict, generation_energy_wms_dict = self._run_generation_phase(
                model, database, runtime_config, batch_size, beam_width, isl, osl, stride
            )

        if latency_correction_scale != 1.0:
            logger.debug(f"latency_correction_scale: {latency_correction_scale} is applied")
            for op in context_latency_dict:
                context_latency_dict[op] *= latency_correction_scale
                context_energy_wms_dict[op] *= latency_correction_scale
            for op in generation_latency_dict:
                generation_latency_dict[op] *= latency_correction_scale
                generation_energy_wms_dict[op] *= latency_correction_scale

        return (
            context_latency_dict,
            context_energy_wms_dict,
            generation_latency_dict,
            generation_energy_wms_dict,
        )

    def run_static_latency_only(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> float:
        """
        Run static inference and return only the total latency in milliseconds.

        This shares the same latency breakdown path as ``run_static`` but skips
        building an ``InferenceSummary``.
        """
        (
            context_latency_dict,
            _,
            generation_latency_dict,
            _,
        ) = self._run_static_breakdown(model, database, runtime_config, mode, stride, latency_correction_scale)
        return sum(context_latency_dict.values()) + sum(generation_latency_dict.values())

    def run_static(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
        afd_config: AfdConfig | None = None,
    ) -> InferenceSummary:
        """
        Run the static inference.

        Args:
            model (BaseModel): the model to run inference
            database (PerfDatabase): the database to run inference
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.
            latency_correction_scale (float): the correction scale to adjust the latency,
                default is 1.0.
                corrected latency = latency * latency_correction_scale
            afd_config (AfdConfig | None): optional heterogeneous AFD configuration.
                When provided, attention and FFN GPU groups can use different
                databases (system YAML), backends, and model configs.
                Fields left as None fall back to the session-level defaults.
        """

        # ---- Resolve heterogeneous AFD resources from AfdConfig ----
        # Each resolved value falls back to the default (model / database) when
        # the corresponding AfdConfig field is None or afd_config itself is None.
        _afd = afd_config
        _afd_attn_db = _afd.attn_database if _afd and _afd.attn_database else database
        _afd_ffn_db = _afd.ffn_database if _afd and _afd.ffn_database else database
        _afd_attn_model = _afd.attn_model if _afd and _afd.attn_model else model
        _afd_ffn_model = _afd.ffn_model if _afd and _afd.ffn_model else model
        _afd_heterogeneous = _afd is not None and (
            _afd.attn_database is not None or _afd.ffn_database is not None
            or _afd.attn_model is not None or _afd.ffn_model is not None
        )

        def _run_context(batch_size: int, isl: int, prefix, ops=None, db=None) -> tuple[dict[str, float], dict[str, float]]:
            """
            Run context phase.

            Args:
                ops: Op list to iterate. Defaults to ``model.context_ops``.
                     Pass ``model.context_attn_ops`` or ``model.context_ffn_ops``
                     for AFD sub-mode runs.
                db: Database to query.  Defaults to the outer *database*
                    parameter.  Override with a per-group database when
                    running heterogeneous AFD.

            Returns:
                tuple: (context_latency_dict, context_energy_wms_dict)
                       latency in ms, energy in W·ms (watt-milliseconds)
            """
            _db = db if db is not None else database

            # (default) If not set, the context will run all the ops
            if ops is None:
                ops = model.context_ops

            context_latency_dict = defaultdict(float)  # milliseconds
            context_energy_wms_dict = defaultdict(float)  # W·ms (watt-milliseconds)

            # isl is corrected based on prefix.
            # Please handle the real logic in your context attention related operations.
            isl = isl - prefix
            if isl <= 0:
                raise ValueError(f"isl must be greater than 0 after removing prefix, but got {isl}")

            for op in ops:
                # query latency and store the latency
                x = batch_size * isl if "logits_gemm" not in op._name else batch_size
                query_kwargs = dict(x=x, batch_size=batch_size, beam_width=1, s=isl, prefix=prefix)

                # Forward AFD kwargs so MoEDispatch.query() can compute P2P comm.
                # Include them for wrapper ops too; OverlapOp forwards kwargs to
                # its inner routed/shared ops.
                if model.config.enable_afd:
                    query_kwargs["num_attn_gpus"] = model.config.num_attn_gpus
                    query_kwargs["num_ffn_gpus"] = model.config.num_ffn_gpus

                result = op.query(_db, **query_kwargs)

                # ✅ IMMEDIATELY extract values - do NOT use PerformanceResult arithmetic!
                latency_ms = float(result)  # Extract latency in milliseconds
                energy_wms = getattr(result, "energy", 0.0)  # Extract energy in watt-milliseconds

                # Aggregate in separate dicts (simple addition)
                context_latency_dict[op._name] += latency_ms
                context_energy_wms_dict[op._name] += energy_wms

            return context_latency_dict, context_energy_wms_dict

        def _run_context_afd_pipelined(
            batch_size: int, isl: int, prefix, num_microbatches: int,
        ) -> tuple[dict[str, float], dict[str, float]]:
            """
            Models the AFD 4-stage pipeline at per-microbatch granularity:

              Stage 1: Attention compute   (on attn GPUs)
              Stage 2: Comm attn → FFN     (P2P / MoE pre-dispatch)
              Stage 3: FFN compute         (on FFN GPUs)
              Stage 4: Comm FFN → attn     (MoE post-dispatch)

            We treat each (microbatch, layer) pair as one task that visits all
            4 stages.  Following the vLLM chunked-prefill pattern (see
            ``vllm_backend._get_mix_step_latency``), each microbatch is
            collapsed to a single combined batch of new tokens:

                mb_bs       = 1
                mb_new_toks = batch_size · (isl - prefix) / M

            With T = M · L tasks flowing through 4 distinct resource-stages,
            the bottleneck stage processes all M microbatches back-to-back
            (cost M · s_max).  Each non-max stage contributes one
            per-(microbatch, layer) cost (= s_i / L) as a one-time fill bubble:

              t_pipe = M · s_max + sum(s_i / L  for i in non-max stages)

            The fill bubble is charged ONCE for the entire run (not once per
            layer), matching MegaScale-Infer / Step-Fun cross-layer pipelining.
            Measuring stages at per-microbatch shape (rather than full shape
            then dividing) captures non-linear scaling of comm / small GEMMs.

            Returns the same (latency_dict, energy_dict) pair as ``_run_context``
            but with total latencies reflecting the pipelined schedule.
            """
            assert num_microbatches == 4, "Currently only consider 4 microbatches to achieve steady state modeling"
            M = max(num_microbatches, 1)
            # Number of transformer layers participating in the pipeline.
            # Both AFD groups should expose the same layer count.
            L = getattr(_afd_attn_model, "_num_layers", None) or getattr(model, "_num_layers", 1)

            # ---- Split work across M microbatches (vLLM-style token chunking) ----
            # Collapse to a single combined batch and slice along the token dim:
            #   mb_bs = 1, mb_new_toks = batch_size * (isl - prefix) / M.
            # Prefix is scaled by the number of full requests in the chunk
            # (matches vllm_backend: ``prefix * floor(ctx_tokens / isl)``).
            eff_isl = isl - prefix
            if eff_isl <= 0:
                raise ValueError(
                    f"isl must be greater than 0 after removing prefix, but got {eff_isl}"
                )
            total_eff_tokens = batch_size * eff_isl
            mb_eff_isl = max(total_eff_tokens // M, 1)
            mb_prefix = int(prefix * (mb_eff_isl // eff_isl))
            # _run_context internally computes effective_isl = isl - prefix,
            # so re-add the (scaled) prefix when forwarding the per-mb isl.
            mb_isl = mb_eff_isl + mb_prefix

            # ---- Measure 4 stages at PER-MICROBATCH shape ----
            # Each measurement is one microbatch's stage cost across all layers.
            # When heterogeneous AFD is active, each stage uses its group's
            # model (for op lists) and database (for perf queries).
            s1_lat, s1_energy = _run_context(
                1, mb_isl, mb_prefix,
                ops=_afd_attn_model.context_attn_compute_ops, db=_afd_attn_db)
            s2_lat, s2_energy = _run_context(
                1, mb_isl, mb_prefix,
                ops=_afd_attn_model.context_comm_a2f_ops, db=_afd_attn_db)
            s3_lat, s3_energy = _run_context(
                1, mb_isl, mb_prefix,
                ops=_afd_ffn_model.context_ffn_compute_ops, db=_afd_ffn_db)
            s4_lat, s4_energy = _run_context(
                1, mb_isl, mb_prefix,
                ops=_afd_ffn_model.context_comm_f2a_ops, db=_afd_ffn_db)

            s = [sum(d.values()) for d in (s1_lat, s2_lat, s3_lat, s4_lat)]
            s_max = max(s)

            # Pipeline formula (per-microbatch granularity):
            #   t_pipe = M · s_max + sum(s_i / L  for i in non-max stages)
            # The first term is the bottleneck stage processing all M microbatches
            # back-to-back.  The second term is the pipeline-fill bubble: each
            # non-max stage contributes one per-microbatch-per-layer cost
            # (= s_i / L) before the bottleneck saturates.
            bubble = sum(si for si in s if si != s_max) / L
            t_pipelined = M * s_max + bubble

            # Total sequential cost across M microbatches (no pipelining).
            # Used only to scale per-op latencies so they sum to t_pipelined.
            total_seq = M * sum(s)
            scale = t_pipelined / total_seq if total_seq > 0 else 1.0

            pipelined_lat = defaultdict(float)
            pipelined_energy = defaultdict(float)
            for stage_lat, stage_energy in [(s1_lat, s1_energy), (s2_lat, s2_energy),
                                            (s3_lat, s3_energy), (s4_lat, s4_energy)]:
                for op_name in stage_lat:
                    # Latency: M microbatches' worth of this op, scaled to t_pipelined.
                    pipelined_lat[op_name] = stage_lat[op_name] * M * scale
                    # Energy is additive across microbatches; pipelining reorders
                    # but does not reduce work.
                    pipelined_energy[op_name] = stage_energy.get(op_name, 0.0) * M

            stage_names = ["attn_compute", "comm_a2f", "ffn_compute", "comm_f2a"]
            logger.debug(
                f"AFD 4-stage pipelined context (per-mb): M={M}, L={L}, "
                f"mb_bs=1, mb_eff_isl={mb_eff_isl}, mb_prefix={mb_prefix} "
                f"(total_eff_tokens={total_eff_tokens}), "
                + ", ".join(f"{n}={v:.3f}ms" for n, v in zip(stage_names, s))
                + f", t_pipelined={t_pipelined:.3f}ms (M·s_max={M*s_max:.3f}ms + bubble={bubble:.3f}ms) "
                + f"vs t_sequential={total_seq:.3f}ms (speedup={total_seq / t_pipelined:.2f}x)"
            )

            return pipelined_lat, pipelined_energy

        def _run_generation_afd_pipelined(
            batch_size: int, beam_width: int, isl: int, osl: int, stride: int,
            num_microbatches: int,
        ) -> tuple[dict[str, float], dict[str, float]]:
            """
            Run generation phase with a 4-stage AFD cross-layer pipeline,
            modelled at per-microbatch granularity.  See
            ``_run_context_afd_pipelined`` for the formula derivation.
            """
            # For generation, the only chunkable dim is batch (1 new token/seq per step).
            # Cap M so each microbatch has at least 1 sequence; if batch_size < requested M,
            # reduce M to batch_size (degenerate M=1 → equivalent to non-pipelined).
            M = max(min(num_microbatches, batch_size), 1)
            L = getattr(_afd_attn_model, "_num_layers", None) or getattr(model, "_num_layers", 1)
            mb_bs = max(batch_size // M, 1)
            s1_lat, s1_energy = _run_generation(
                mb_bs, beam_width, isl, osl, stride,
                ops=_afd_attn_model.generation_attn_compute_ops, db=_afd_attn_db)
            s2_lat, s2_energy = _run_generation(
                mb_bs, beam_width, isl, osl, stride,
                ops=_afd_attn_model.generation_comm_a2f_ops, db=_afd_attn_db)
            s3_lat, s3_energy = _run_generation(
                mb_bs, beam_width, isl, osl, stride,
                ops=_afd_ffn_model.generation_ffn_compute_ops, db=_afd_ffn_db)
            s4_lat, s4_energy = _run_generation(
                mb_bs, beam_width, isl, osl, stride,
                ops=_afd_ffn_model.generation_comm_f2a_ops, db=_afd_ffn_db)

            s = [sum(d.values()) for d in (s1_lat, s2_lat, s3_lat, s4_lat)]
            s_max = max(s)

            # Pipeline formula (per-microbatch granularity):
            #   t_pipe = M · s_max + sum(s_i / L  for i in non-max stages)
            bubble = sum(si for si in s if si != s_max) / L
            t_pipelined = M * s_max + bubble

            total_seq = M * sum(s)
            scale = t_pipelined / total_seq if total_seq > 0 else 1.0

            pipelined_lat = defaultdict(float)
            pipelined_energy = defaultdict(float)
            for stage_lat, stage_energy in [(s1_lat, s1_energy), (s2_lat, s2_energy),
                                            (s3_lat, s3_energy), (s4_lat, s4_energy)]:
                for op_name in stage_lat:
                    pipelined_lat[op_name] = stage_lat[op_name] * M * scale
                    pipelined_energy[op_name] = stage_energy.get(op_name, 0.0) * M

            stage_names = ["attn_compute", "comm_a2f", "ffn_compute", "comm_f2a"]
            logger.debug(
                f"AFD 4-stage pipelined generation (per-mb): M={M}, L={L}, mb_bs={mb_bs}, "
                + ", ".join(f"{n}={v:.3f}ms" for n, v in zip(stage_names, s))
                + f", t_pipelined={t_pipelined:.3f}ms (M·s_max={M*s_max:.3f}ms + bubble={bubble:.3f}ms) "
                + f"vs t_sequential={total_seq:.3f}ms"
            )

            return pipelined_lat, pipelined_energy

        def _run_generation(
            batch_size: int, beam_width: int, isl: int, osl: int, stride: int, ops=None, db=None,
        ) -> tuple[dict[str, float], dict[str, float]]:
            """
            Run generation phase.

            Args:
                ops: Op list to iterate. Defaults to ``model.generation_ops``.
                     Pass ``model.generation_attn_ops`` or ``model.generation_ffn_ops``
                     for AFD sub-mode runs.
                db: Database to query.  Defaults to the outer *database*
                    parameter.  Override with a per-group database when
                    running heterogeneous AFD.

            Returns:
                tuple: (generation_latency_dict, generation_energy_wms_dict)
                       latency in ms, energy in W·ms
            """
            _db = db if db is not None else database

            if ops is None:
                ops = model.generation_ops

            # mtp/speculative decoding correction
            batch_size = batch_size * (model._nextn + 1)

            generation_latency_dict = defaultdict(float)  # milliseconds
            generation_energy_wms_dict = defaultdict(float)  # W·ms

            for i in range(0, osl - 1, stride):
                latency_dict = defaultdict(float)
                energy_wms_dict = defaultdict(float)  # W·ms

                for op in ops:
                    query_kwargs = dict(
                        x=batch_size * beam_width,
                        batch_size=batch_size,
                        beam_width=beam_width,
                        s=isl + i + 1,
                    )

                    # Forward AFD kwargs so MoEDispatch.query() can compute P2P comm.
                    # Include them for wrapper ops too; OverlapOp forwards kwargs to
                    # its inner routed/shared ops.
                    if model.config.enable_afd:
                        query_kwargs["num_attn_gpus"] = model.config.num_attn_gpus
                        query_kwargs["num_ffn_gpus"] = model.config.num_ffn_gpus

                    result = op.query(_db, **query_kwargs)

                    # ✅ IMMEDIATELY extract values - do NOT accumulate PerformanceResult objects!
                    latency_ms = float(result)
                    energy_wms = getattr(result, "energy", 0.0)

                    latency_dict[op._name] += latency_ms
                    energy_wms_dict[op._name] += energy_wms

                # usually stride, but might be less at the end
                repeat_count = min(stride, osl - 1 - i)

                for op in latency_dict:
                    # Both latency and energy are additive - multiply by repeat_count
                    generation_latency_dict[op] += latency_dict[op] * repeat_count
                    generation_energy_wms_dict[op] += energy_wms_dict[op] * repeat_count  # SIMPLIFIED

            return generation_latency_dict, generation_energy_wms_dict

        summary = InferenceSummary(runtime_config)
        batch_size, beam_width, isl, osl, prefix = (
            runtime_config.batch_size,
            runtime_config.beam_width,
            runtime_config.isl,
            runtime_config.osl,
            runtime_config.prefix,
        )

        # ----- AFD (Attention-FFN Disaggregation) support -----
        
        # Match the mode to the ops list in context/generation phases
        _afd_ctx_op_lists = {
            "static_ctx_attn": "context_attn_ops",
            "static_ctx_ffn": "context_ffn_ops",
        }
        _afd_gen_op_lists = {
            "static_gen_attn": "generation_attn_ops",
            "static_gen_ffn": "generation_ffn_ops",
        }

        # Check if AFD pipelining is enabled
        _afd_pipeline = (
            model.config.enable_afd
            and getattr(model.config, 'afd_num_microbatches', 1) > 1
        )
        _afd_M = getattr(model.config, 'afd_num_microbatches', 1) if _afd_pipeline else 1

        # Helper to pick AFD-aware or standard memory calculation
        _use_afd_memory = (
            model.config.enable_afd
            and model.config.num_attn_gpus is not None
            and hasattr(self, '_get_afd_memory_usage')
        )

        def _get_memory(bs, bw, _isl, _osl, _num_tokens=0):
            if _use_afd_memory:
                return self._get_afd_memory_usage(model, database, bs, bw, _isl, _osl, _num_tokens)
            return self._get_memory_usage(model, database, bs, bw, _isl, _osl, _num_tokens)

        # --- Resolve per-group ops source model and database for AFD sub-modes ---
        # For heterogeneous AFD, attn sub-modes use _afd_attn_model ops queried
        # against _afd_attn_db, and ffn sub-modes use _afd_ffn_model / _afd_ffn_db.
        _afd_ctx_ops_db = {
            "static_ctx_attn": (_afd_attn_model, "context_attn_ops", _afd_attn_db),
            "static_ctx_ffn":  (_afd_ffn_model,  "context_ffn_ops",  _afd_ffn_db),
        }
        _afd_gen_ops_db = {
            "static_gen_attn": (_afd_attn_model, "generation_attn_ops", _afd_attn_db),
            "static_gen_ffn":  (_afd_ffn_model,  "generation_ffn_ops",  _afd_ffn_db),
        }

        context_latency_dict = defaultdict(float)
        context_energy_wms_dict = defaultdict(float)
        generation_latency_dict = defaultdict(float)
        generation_energy_wms_dict = defaultdict(float)

        if mode in _afd_ctx_ops_db:
            _m, _attr, _db = _afd_ctx_ops_db[mode]
            afd_ops = getattr(_m, _attr)
            context_latency_dict, context_energy_wms_dict = _run_context(
                batch_size, isl, prefix, ops=afd_ops, db=_db,
            )
            memory = _get_memory(batch_size, beam_width, isl, 1)
        elif mode in _afd_gen_ops_db:
            _m, _attr, _db = _afd_gen_ops_db[mode]
            afd_ops = getattr(_m, _attr)
            generation_latency_dict, generation_energy_wms_dict = _run_generation(
                batch_size, beam_width, isl, osl, stride, ops=afd_ops, db=_db,
            )
            memory = _get_memory(
                batch_size, beam_width, isl, osl, _num_tokens=batch_size * beam_width,
            )
        elif mode == "static_ctx":
            if _afd_pipeline:
                context_latency_dict, context_energy_wms_dict = _run_context_afd_pipelined(
                    batch_size, isl, prefix, num_microbatches=_afd_M,
                )
            else:
                context_latency_dict, context_energy_wms_dict = _run_context(batch_size, isl, prefix)
            memory = _get_memory(batch_size, beam_width, isl, 1)
        elif mode == "static_gen":
            if _afd_pipeline:
                generation_latency_dict, generation_energy_wms_dict = _run_generation_afd_pipelined(
                    batch_size, beam_width, isl, osl, stride, num_microbatches=_afd_M,
                )
            else:
                generation_latency_dict, generation_energy_wms_dict = _run_generation(
                    batch_size, beam_width, isl, osl, stride
                )
            memory = _get_memory(
                batch_size, beam_width, isl, osl, _num_tokens=batch_size * beam_width,
            )  # for gen only, all kvcache is needed.
        else:
            if _afd_pipeline:
                context_latency_dict, context_energy_wms_dict = _run_context_afd_pipelined(
                    batch_size, isl, prefix, num_microbatches=_afd_M,
                )
                generation_latency_dict, generation_energy_wms_dict = _run_generation_afd_pipelined(
                    batch_size, beam_width, isl, osl, stride, num_microbatches=_afd_M,
                )
            else:
                context_latency_dict, context_energy_wms_dict = _run_context(batch_size, isl, prefix)
                generation_latency_dict, generation_energy_wms_dict = _run_generation(
                    batch_size, beam_width, isl, osl, stride
                )
            memory = _get_memory(batch_size, beam_width, isl, osl)

        if latency_correction_scale != 1.0:
            logger.debug(f"latency_correction_scale: {latency_correction_scale} is applied")
            for op in context_latency_dict:
                context_latency_dict[op] *= latency_correction_scale
                context_energy_wms_dict[op] *= latency_correction_scale  # Energy scales with latency!
            for op in generation_latency_dict:
                generation_latency_dict[op] *= latency_correction_scale
                generation_energy_wms_dict[op] *= latency_correction_scale  # Energy scales with latency!

        # Calculate total latencies and energies (simple sums - decoupled!)
        context_latency_ms = sum(context_latency_dict.values())  # milliseconds
        context_energy_wms = sum(context_energy_wms_dict.values())  # watt-milliseconds

        generation_latency_ms = sum(generation_latency_dict.values())  # milliseconds
        generation_energy_wms = sum(generation_energy_wms_dict.values())  # watt-milliseconds

        # Calculate average power (SIMPLIFIED - just divide! Single operation.)
        context_power_avg = context_energy_wms / context_latency_ms if context_latency_ms > 0 else 0.0
        generation_power_avg = generation_energy_wms / generation_latency_ms if generation_latency_ms > 0 else 0.0

        # E2E weighted average power (EVEN SIMPLER - natural weighted average!)
        total_latency_ms = context_latency_ms + generation_latency_ms
        total_energy_wms = context_energy_wms + generation_energy_wms
        e2e_power_avg = total_energy_wms / total_latency_ms if total_latency_ms > 0 else 0.0

        # For backward compatibility, keep old variable names
        context_latency = context_latency_ms
        generation_latency = generation_latency_ms

        bs = batch_size
        global_bs = bs * model.config.attention_dp_size
        concurrency = global_bs
        ttft = context_latency
        tpot = 0.0 if osl <= 1 else generation_latency / (osl - 1)
        num_generated_tokens = max(osl - 1, 0)
        request_latency = ttft + tpot * num_generated_tokens
        if request_latency == 0.0:
            request_latency = context_latency + generation_latency
        request_rate = 0.0
        tp = model.config.tp_size
        pp = model.config.pp_size
        dp = model.config.attention_dp_size
        moe_tp = model.config.moe_tp_size
        moe_ep = model.config.moe_ep_size

        # In AFD mode, attention and FFN GPUs are decoupled physical groups.
        # num_total_gpus = num_attn_gpus + num_ffn_gpus (not tp*pp*dp).
        if model.config.enable_afd and model.config.num_attn_gpus is not None:
            num_total_gpus = model.config.num_attn_gpus + model.config.num_ffn_gpus
        else:
            num_total_gpus = tp * pp * dp

        seq_s = (
            0.0 if request_latency == 0.0 else global_bs / request_latency * 1000 * model.config.pp_size
        )  # handle statc_gen only with osl==1, scale by pp
        seq_s_gpu = seq_s / num_total_gpus
        tokens_s = seq_s * osl if mode != "static_gen" else seq_s * (osl - 1)
        if mode == "static_ctx":
            tokens_s = seq_s * 1  # only first token
        tokens_s_gpu = tokens_s / num_total_gpus
        tokens_s_user = 0.0 if tpot == 0.0 else 1000.0 / tpot
        parallel = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}"
        gemm = model.config.gemm_quant_mode.name
        kvcache = model.config.kvcache_quant_mode.name
        fmha = model.config.fmha_quant_mode.name
        moe = model.config.moe_quant_mode.name
        comm = model.config.comm_quant_mode.name
        mem = memory["total"]

        # AFD columns (None when not in AFD mode)
        num_attn_gpus = model.config.num_attn_gpus
        num_ffn_gpus = model.config.num_ffn_gpus

        data = [
            [
                model.model_path,
                isl,
                osl,
                prefix,
                concurrency,
                request_rate,
                bs,
                global_bs,
                ttft,
                tpot,
                seq_s,
                seq_s_gpu,
                tokens_s,
                tokens_s_gpu,
                tokens_s_user,
                request_latency,
                context_latency,
                generation_latency,
                num_total_gpus,
                num_attn_gpus,
                num_ffn_gpus,
                tp,
                pp,
                dp,
                moe_tp,
                moe_ep,
                parallel,
                gemm,
                kvcache,
                fmha,
                moe,
                comm,
                mem,
                database.backend,
                database.version,
                database.system,
                e2e_power_avg,  # NEW: E2E weighted average power in watts
            ]
        ]

        summary_df = pd.DataFrame(data, columns=common.ColumnsStatic).round(3)

        summary.set_context_latency_dict(context_latency_dict)
        summary.set_generation_latency_dict(generation_latency_dict)
        summary.set_context_energy_wms_dict(context_energy_wms_dict)  # UPDATED: explicit units
        summary.set_generation_energy_wms_dict(generation_energy_wms_dict)  # UPDATED: explicit units
        summary.set_context_power_avg(context_power_avg)
        summary.set_generation_power_avg(generation_power_avg)
        summary.set_e2e_power_avg(e2e_power_avg)

        # --- OOM check ---
        # For heterogeneous AFD, each GPU group may have a different memory
        # capacity (different system YAML).  Check each group against its own
        # capacity when the memory dict contains per-group breakdowns.
        if _afd_heterogeneous and _use_afd_memory:
            attn_cap = _afd_attn_db.system_spec["gpu"]["mem_capacity"]
            ffn_cap = _afd_ffn_db.system_spec["gpu"]["mem_capacity"]
            attn_total_gb = memory.get("attn_total", 0.0)
            ffn_total_gb = memory.get("ffn_total", 0.0)
            is_oom = (
                attn_total_gb >= (attn_cap / (1 << 30))
                or ffn_total_gb >= (ffn_cap / (1 << 30))
            )
            # Still call set_memory_and_check_oom for the dict, but override
            # the OOM decision using the max capacity for backward compat.
            summary.set_memory_and_check_oom(
                memory, max(attn_cap, ffn_cap)
            )
            summary.set_oom(is_oom)
        else:
            summary.set_memory_and_check_oom(memory, database.system_spec["gpu"]["mem_capacity"])

        summary.set_summary_df(summary_df)

        return summary

    def get_kv_cache_size_bytes_per_rank(
        self,
        model: BaseModel,
        batch_size: int,
        isl: int,
        beam_width: int = 1,
        osl: int = 0,
    ) -> int:
        """
        Return per-rank resident KV cache size in bytes.

        This helper is intended for per-GPU memory accounting. For TP-sharded
        models it returns the KV bytes owned by a single TP rank, not the total
        logical KV bytes for the request.
        """
        if model.model_family in ("DEEPSEEK", "DEEPSEEKV32"):
            kvcache_per_token = model._num_layers * 576
        else:
            num_kv_heads_per_gpu = (model._num_kv_heads + model.config.tp_size - 1) // model.config.tp_size
            kvcache_per_token = num_kv_heads_per_gpu * model._head_size * model._num_layers * 2

        return (
            (batch_size * isl + batch_size * beam_width * osl)
            * model.config.kvcache_quant_mode.value.memory
            * kvcache_per_token
        )

    def get_kv_cache_size_bytes(
        self,
        model: BaseModel,
        batch_size: int,
        isl: int,
        beam_width: int = 1,
        osl: int = 0,
    ) -> int:
        """
        Backward-compatible alias for ``get_kv_cache_size_bytes_per_rank()``.
        """
        return self.get_kv_cache_size_bytes_per_rank(
            model=model,
            batch_size=batch_size,
            isl=isl,
            beam_width=beam_width,
            osl=osl,
        )

    def get_total_kv_cache_transfer_size_bytes(
        self,
        model: BaseModel,
        batch_size: int,
        isl: int,
        beam_width: int = 1,
        osl: int = 0,
    ) -> int:
        """
        Return total logical KV cache bytes for network-transfer modeling.

        Unlike ``get_kv_cache_size_bytes_per_rank()``, this helper does not
        shard KV by TP rank. The returned value represents the full KV state
        for the request at the prefill/decode boundary, so callers can
        distribute or reshape it across TP ranks without double-dividing by TP.
        """
        if model.model_family in ("DEEPSEEK", "DEEPSEEKV32"):
            # DeepSeek MLA uses a separate compressed KV representation. The
            # current model treats the 576-byte latent as the logical per-layer
            # cache unit, so keep transfer sizing aligned with the existing MLA
            # memory model until per-rank MLA KV ownership is modeled explicitly.
            kvcache_per_token = model._num_layers * 576
        else:
            kvcache_per_token = model._num_kv_heads * model._head_size * model._num_layers * 2

        return (
            (batch_size * isl + batch_size * beam_width * osl)
            * model.config.kvcache_quant_mode.value.memory
            * kvcache_per_token
        )

    def _get_ctx_tokens_list_for_agg_sweep(
        self,
        isl: int,
        ctx_stride: int,
        enable_chunked_prefill: bool,
        max_normal_ctx_tokens: int = 8192,
        max_ctx_tokens_multiple_of_isl: int = 2,
        max_ctx_tokens_small_search_steps: int = 16,
        max_ctx_tokens_search_steps: int = 8,
    ) -> list[int]:
        """
        Generate a list of num_context_tokens to sweep for agg inference.

        Args:
            isl: Target input sequence length during inference.
            ctx_stride: Default stride for context_tokens to sweep, ignored if enable_chunked_prefill is True.
            enable_chunked_prefill: Whether the inference framework will have chunked_prefill enabled.
            max_normal_ctx_tokens: boundary at which to increase the stride for faster sweeping.
            max_ctx_tokens_multiple_of_isl: Maximum multiple of isl to consider for ctx tokens.
            max_ctx_tokens_small_search_steps: Maximum search steps under max_normal_ctx_tokens.
            max_ctx_tokens_large_search_steps: Maximum search steps over max_normal_ctx_tokens.
        Returns:
            Sorted list of num_context_tokens to sweep.
        """

        # Largest ctx_tokens to consider for sweeping.
        max_ctx_tokens = max(max_normal_ctx_tokens, isl * max_ctx_tokens_multiple_of_isl)

        # Sweep stride under max_normal_ctx_tokens.
        ctx_stride = max(ctx_stride, max_normal_ctx_tokens // max_ctx_tokens_small_search_steps)

        # Sweep stride once ctx_tokens is larger than max_normal_ctx_tokens.
        ctx_stride_large = max(
            1024,
            ctx_stride,
            max_ctx_tokens // max_ctx_tokens_search_steps,
        )

        if not enable_chunked_prefill:
            new_ctx_stride = max(isl, ctx_stride)
            new_ctx_stride_large = int(np.ceil(ctx_stride_large / isl) * isl)
            logger.debug(
                f"enable_chunked_prefill is off, override ctx_stride: from {ctx_stride} to {new_ctx_stride}, "
                f"ctx_stride_large: from {ctx_stride_large} to {new_ctx_stride_large}"
            )
            ctx_stride = new_ctx_stride
            ctx_stride_large = new_ctx_stride_large

        # prepare ctx_tokens_list
        ctx_tokens_list = []
        ctx_tokens = 0
        while True:
            if ctx_tokens < max_normal_ctx_tokens:
                ctx_tokens += ctx_stride
            else:
                ctx_tokens += ctx_stride_large

            if ctx_tokens > max_ctx_tokens:
                break

            ctx_tokens_list.append(ctx_tokens)

        # add those just match the multiple of isl
        for i in range(1, max_ctx_tokens_multiple_of_isl + 1):
            ctx_tokens = isl * i
            if ctx_tokens not in ctx_tokens_list:
                ctx_tokens_list.append(ctx_tokens)
        ctx_tokens_list.sort()
        return ctx_tokens_list

    @abstractmethod
    def run_agg(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Run the agg inference.
        """
        pass

    @abstractmethod
    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints.
        """
        pass

    @abstractmethod
    def _get_memory_usage(
        self,
        model: BaseModel,
        database: PerfDatabase,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
        prefix: int = 0,
    ) -> dict[str, float]:
        """
        Get the memory usage of the backend.

        Args:
            prefix: number of prefix tokens (part of isl) whose KV is already cached
                (per-request) and does not need activation computation.
        """
        pass
