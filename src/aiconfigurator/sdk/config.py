# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from aiconfigurator.sdk import common

if TYPE_CHECKING:
    from aiconfigurator.sdk.backends.base_backend import BaseBackend
    from aiconfigurator.sdk.models import BaseModel
    from aiconfigurator.sdk.perf_database import PerfDatabase


@dataclass
class ModelConfig:
    """
    Model configuration.
    """

    tp_size: int = 1
    pp_size: int = 1
    gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.float16
    moe_quant_mode: common.MoEQuantMode = common.MoEQuantMode.float16
    kvcache_quant_mode: common.KVCacheQuantMode = common.KVCacheQuantMode.float16
    fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.float16
    comm_quant_mode: common.CommQuantMode = common.CommQuantMode.half
    moe_tp_size: int = None
    moe_ep_size: int = None
    attention_dp_size: int = 1
    workload_distribution: str = "power_law"
    nextn: int = 0  # at most mtp5
    nextn_accept_rates: list = None
    overwrite_num_layers: int = 0
    # model builder falvors
    sms: int = 20
    moe_backend: str = None  # for sglang wideep only, deepep
    attention_backend: str = "flashinfer"  # 'flashinfer' or 'fa3', for sglang wideep only
    enable_wideep: bool = False
    enable_afd: bool = False  # enable AFD (Attention-FFN Disaggregation) inter-machine communication
    num_attn_gpus: int = None  # AFD: number of GPUs dedicated to attention (decoupled from FFN)
    num_ffn_gpus: int = None  # AFD: number of GPUs dedicated to FFN/MoE (decoupled from attention)
    afd_num_microbatches: int = 1  # AFD: number of microbatches for 4-stage pipeline (1 = no overlap, max 4)


@dataclass
class RuntimeConfig:
    """
    Runtime configuration.
    """

    batch_size: int = None
    beam_width: int = 1
    isl: int = None
    osl: int = None
    prefix: int = 0  # prefix len of isl
    ttft: float = None
    tpot: Union[float, list] = None
    request_latency: float = None  # it works together with ttft. 1. <= req_lat 2. <= req_lat and <= ttft


@dataclass
class AfdConfig:
    """
    Optional configuration for heterogeneous AFD (Attention-FFN Disaggregation).

    When provided to ``InferenceSession`` or ``DisaggInferenceSession``, allows
    attention and FFN GPU groups to use **different** performance databases (i.e.
    different system YAML files / hardware), backends, and model configurations
    (e.g. different TP sizes or quantization modes).

    This mirrors the prefill / decode split already available in
    ``DisaggInferenceSession``, but applied *within* a single worker to the
    attention vs. FFN partition.

    **Fallback behaviour** — any field left as ``None`` inherits the session's
    default model, database, or backend.

    Typical usage::

        # Attention GPUs on H100-SXM, FFN GPUs on B200-SXM
        afd = AfdConfig(
            attn_database=h100_db,
            ffn_database=b200_db,
        )
        sess = InferenceSession(model=model, database=h100_db,
                                backend=backend, afd_config=afd)

    For full heterogeneity (different TP / quant per group), also set
    ``attn_model`` and ``ffn_model`` which should be built via
    ``models.get_model()`` with the appropriate ``ModelConfig``.
    """

    # --- Attention GPU group ---
    attn_model: BaseModel | None = None
    attn_database: PerfDatabase | None = None
    attn_backend: BaseBackend | None = None

    # --- FFN GPU group ---
    ffn_model: BaseModel | None = None
    ffn_database: PerfDatabase | None = None
    ffn_backend: BaseBackend | None = None
