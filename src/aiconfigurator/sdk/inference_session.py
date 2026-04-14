# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import logging
import math
import warnings

import pandas as pd

from aiconfigurator.sdk import common, config, models, perf_database
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints

# Import AstraSim from astrasim_utils, which defines imports and initialization in one place
from aiconfigurator.sdk import astrasim_utils
from aiconfigurator.sdk.astrasim_utils import AstraSimManager

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)

# AstraSim availability flag (used for backward compat checks)
NETWORK_SIM_AVAILABLE = astrasim_utils.NETWORK_SIM_AVAILABLE


class InferenceSession:
    """
    InferenceSession holds the model and database to run inference loop

    Attributes:
        model (models.BaseModel): the model to run inference
        database (perf_database.PerfDatabase): the database to run inference
        backend (backend.Backend): the backend to run inference

    Methods:
        run_static (static, static_ctx, static_gen): to support static batching and disagg,
            returns details of a static run
        run_agg (static, static_ctx, static_gen): run agg inference, returns summary of the
            perf result with given agg config and runtime config (concurrency)
        find_best_agg_result_under_constraints (static, static_ctx, static_gen):
            find the best agg result under constraints, returns summary
            which contains all the possible agg config and perf that matchs SLA.
    """

    def __init__(
        self,
        model: models.BaseModel,
        database: perf_database.PerfDatabase,
        backend: BaseBackend,
        afd_config: config.AfdConfig | None = None,
    ) -> None:
        """
        Initialize the InferenceSession

        Args:
            model: The model to run inference.
            database: The performance database.
            backend: The backend (TRT-LLM, vLLM, SGLang, …).
            afd_config: Optional heterogeneous AFD configuration.
                When provided, attention and FFN GPU groups can use
                different databases (system YAML), backends, and model
                configs.  Fields left as ``None`` fall back to the
                session-level defaults.
        """
        self._model = model
        self._database = database
        self._backend = backend
        self._afd_config = afd_config

    def run_static(
        self,
        runtime_config: config.RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> InferenceSummary:
        """
        Run static inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_static(
            self._model, self._database, runtime_config, mode, stride, latency_correction_scale,
            afd_config=self._afd_config,
        )

    def run_agg(self, runtime_config: config.RuntimeConfig, **kwargs) -> InferenceSummary:
        """
        Run agg inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to run agg, depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_agg(self._model, self._database, runtime_config, afd_config=self._afd_config, **kwargs)

    # Optimization
    def find_best_agg_result_under_constraints(
        self, runtime_config: config.RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to find the best agg result under constraints,
                depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result, contains all the possible
                agg config and perf that matchs SLA.
        """
        return self._backend.find_best_agg_result_under_constraints(
            self._model, self._database, runtime_config, afd_config=self._afd_config, **kwargs
        )


DECODE_FILTER_RATIO_MIN = 0.0
DECODE_FILTER_RATIO_MAX = 1.0
MAX_DECODE_WORKERS_PER_CATEGORY = 16
MAX_PREFILL_WORKERS = 32
MAX_NUM_DECODE_WORKER_CANDIDATES = 64
MAX_NUM_PREFILL_WORKER_CANDIDATES = 32
VALID_GPU_LAYOUT_STRATEGIES = (
    "segregated_by_phase",
    "paired_prefill_decode_per_node",
)


class DisaggInferenceSession:
    """
    Disaggregated inference session
    Run prefill and generation separately, with different models (parallel and precision config can
    be different) and databases
    0. init func only takes database and backend, model is passed in run_disagg
    1. run_disagg, given model, database and backend, given everything fixed ((max)batchsize and
       num_workers) , return the perf result of the system
    2. find_best_disagg_result_under_constraints, given database and backend, sweep batchsize and
       model parallel to match SLA, sweep workers to get best system perf/gpu if allowed.
       Return config (parallel, batchsize and num_workers) and perf.
    3. TODO, should consider kvcache model in future
    Disagg is more like a post processing step to do rate matching, that's why it's a
    DiaggInferenceSession instread of using InferenceSession.

    Attributes:
        prefill_database (perf_database.PerfDatabase): the database to run prefill
        prefill_backend (backend.Backend): the backend to run prefill
        decode_database (perf_database.PerfDatabase): the database to run decode
        decode_backend (backend.Backend): the backend to run decode

    Methods:
        run_disagg (model_path, runtime_config, prefill_model_config, prefill_batch_size,
                    prefill_num_worker, decode_model_config, decode_batch_size,
                    decode_num_worker)
            run disagg with given prefill/decode worker info
        find_best_disagg_result_under_constraints (model_path,runtime_config, prefill_model_config,
                    prefill_parallel_config_list, prefill_max_num_tokens, prefill_num_worker_list,
                    decode_model_config, decode_parallel_config_list, decode_max_num_tokens,
                    decode_num_worker_list, num_gpu_list)
            find the best disagg result under constraints
        set_latency_correction_scales (prefill_latency_correction_scale,
                                       decode_latency_correction_scale):
            set the correction scales for better alignment with real system
    """

    def __init__(
        self,
        prefill_database: perf_database.PerfDatabase,
        prefill_backend: BaseBackend,
        decode_database: perf_database.PerfDatabase,
        decode_backend: BaseBackend,
        network_file: str = None,  # optional explicit network config file
        astrasim_manager: AstraSimManager | None = None,  # preferred: pass a manager
        gpu_layout_strategy: str = "segregated_by_phase",
        prefill_afd_config: config.AfdConfig | None = None,
        decode_afd_config: config.AfdConfig | None = None,
    ) -> None:
        """
        Initialize the DisaggInferenceSession

        Args:
            prefill_database: Performance database for prefill workers.
            prefill_backend: Backend for prefill workers.
            decode_database: Performance database for decode workers.
            decode_backend: Backend for decode workers.
            network_file: Explicit path to an AstraSim topology YAML.
                Ignored when *astrasim_manager* is provided.
            astrasim_manager: Pre-configured :class:`AstraSimManager`.
                When ``None``, one is created automatically (using
                *network_file* if given, otherwise auto-deriving from
                the prefill database's system spec).
            gpu_layout_strategy: How logical prefill/decode workers are mapped
                onto GPU IDs for network-placement studies.
            prefill_afd_config: Optional heterogeneous AFD configuration for
                prefill workers.  When provided, the inner prefill
                ``InferenceSession`` uses different databases / models
                for attention vs FFN GPU groups.
            decode_afd_config: Optional heterogeneous AFD configuration for
                decode workers.  Same semantics as *prefill_afd_config*
                but applied to decode workers.
        """
        if gpu_layout_strategy not in VALID_GPU_LAYOUT_STRATEGIES:
            raise ValueError(
                f"Invalid gpu_layout_strategy '{gpu_layout_strategy}'. "
                f"Must be one of {VALID_GPU_LAYOUT_STRATEGIES}"
            )
        self._prefill_database = prefill_database
        self._prefill_backend = prefill_backend
        self._decode_database = decode_database
        self._decode_backend = decode_backend
        self._gpu_layout_strategy = gpu_layout_strategy

        # Heterogeneous AFD configs for prefill / decode workers
        self._prefill_afd_config = prefill_afd_config
        self._decode_afd_config = decode_afd_config

        # allow user to set correction scales for better alignment with real system
        self._prefill_latency_correction_scale = 1.0
        self._decode_latency_correction_scale = 1.0

        # comes from pipeline bubble, especially when benchmarking with concurrency
        self._RATE_MATCHING_PREFILL_DEGRADATION_FACTOR = 0.9
        # comes from not saturating the batchsize slot of decode worker
        self._RATE_MATCHING_DECODE_DEGRADATION_FACTOR = 0.92

        # AstraSim network simulation via AstraSimManager
        if astrasim_manager is not None:
            self._astrasim = astrasim_manager
        elif network_file is not None:
            # Explicit config file → manager with that file
            self._astrasim = AstraSimManager(
                network_config=network_file,
            )
        else:
            # Auto-derive from the prefill database's system spec
            system_spec = getattr(prefill_database, "system_spec", None)
            self._astrasim = AstraSimManager(
                system_spec=system_spec,
            )

        self._enable_astrasim = self._astrasim.enabled
        if self._enable_astrasim:
            logger.info("AstraSim Network Engine enabled for disagg KV cache transfer")

    def set_latency_correction_scales(
        self, prefill_latency_correction_scale: float, decode_latency_correction_scale: float
    ):
        """
        Set the correction scales for better alignment with real system
        """
        self._prefill_latency_correction_scale = prefill_latency_correction_scale
        self._decode_latency_correction_scale = decode_latency_correction_scale

    def _get_disagg_summary_dict(
        self,
        prefill_summary_dict: dict,
        prefill_num_worker: int,
        decode_summary_dict: dict,
        decode_num_worker: int,
    ) -> dict:
        """
        Get the disagg summary as a dict based on prefill and decode summary dicts.
        The summary dict is used for efficient batch operations.
        """
        seq_s = min(
            prefill_summary_dict["seq/s"] * prefill_num_worker * self._RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
            decode_summary_dict["seq/s"] * decode_num_worker * self._RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
        )
        # Use num_total_gpus directly — already accounts for AFD decoupled GPUs
        prefill_gpus = prefill_summary_dict["num_total_gpus"]
        decode_gpus = decode_summary_dict["num_total_gpus"]
        seq_s_gpu = seq_s / (prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker)

        tokens_s = seq_s * prefill_summary_dict["osl"]
        tokens_s_gpu = tokens_s / (prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker)
        num_total_gpus = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker
        osl = prefill_summary_dict["osl"]
        request_latency = prefill_summary_dict["ttft"] + decode_summary_dict["tpot"] * max(osl - 1, 0)

        # Calculate weighted average power for DISAGG mode
        # Power is weighted by time spent in each phase
        # Note: prefill_power and decode_power are already per-GPU averages
        ttft = prefill_summary_dict["ttft"]
        tpot = decode_summary_dict["tpot"]
        decode_time = tpot * max(osl - 1, 0)

        prefill_power = prefill_summary_dict.get("power_w", 0.0)
        decode_power = decode_summary_dict.get("power_w", 0.0)

        # DEBUG: Log the power values we're getting
        logger.debug(
            f"DISAGG Power Calc: prefill_power={prefill_power}W, "
            f"decode_power={decode_power}W, ttft={ttft}ms, decode_time={decode_time}ms"
        )

        # Simple time-weighted average (power values are already per-GPU)
        total_time = ttft + decode_time

        if total_time > 0:
            disagg_power_avg = (prefill_power * ttft + decode_power * decode_time) / total_time
        else:
            disagg_power_avg = 0.0

        logger.debug(
            f"DISAGG Power Result: {disagg_power_avg}W (time-weighted from {prefill_power}W and {decode_power}W)"
        )

        return {
            "model": prefill_summary_dict["model"],
            "isl": prefill_summary_dict["isl"],
            "osl": prefill_summary_dict["osl"],
            "prefix": prefill_summary_dict["prefix"],
            # This is not exact matching. You can use this concurrency to benchmark the system.
            "concurrency": decode_summary_dict["concurrency"] * decode_num_worker,
            "request_rate": seq_s,
            "(p)bs": prefill_summary_dict["bs"],
            "(p)global_bs": prefill_summary_dict["global_bs"],
            "(p)workers": prefill_num_worker,
            "(d)bs": decode_summary_dict["bs"],
            "(d)global_bs": decode_summary_dict["global_bs"],
            "(d)workers": decode_num_worker,
            "ttft": prefill_summary_dict["ttft"],
            "tpot": decode_summary_dict["tpot"],
            "request_latency": request_latency,
            "seq/s": seq_s,
            "seq/s/gpu": seq_s_gpu,
            "tokens/s": tokens_s,
            "tokens/s/gpu": tokens_s_gpu,
            "tokens/s/user": decode_summary_dict["tokens/s/user"],
            "(p)seq/s/worker": prefill_summary_dict["seq/s"],
            "(d)seq/s/worker": decode_summary_dict["seq/s"],
            "num_total_gpus": num_total_gpus,
            "(p)num_attn_gpus": prefill_summary_dict.get("num_attn_gpus"),
            "(p)num_ffn_gpus": prefill_summary_dict.get("num_ffn_gpus"),
            "(d)num_attn_gpus": decode_summary_dict.get("num_attn_gpus"),
            "(d)num_ffn_gpus": decode_summary_dict.get("num_ffn_gpus"),
            "(p)tp": prefill_summary_dict["tp"],
            "(p)pp": prefill_summary_dict["pp"],
            "(p)dp": prefill_summary_dict["dp"],
            "(p)moe_tp": prefill_summary_dict["moe_tp"],
            "(p)moe_ep": prefill_summary_dict["moe_ep"],
            "(p)parallel": prefill_summary_dict["parallel"],
            "(p)gemm": prefill_summary_dict["gemm"],
            "(p)kvcache": prefill_summary_dict["kvcache"],
            "(p)fmha": prefill_summary_dict["fmha"],
            "(p)moe": prefill_summary_dict["moe"],
            "(p)comm": prefill_summary_dict["comm"],
            "(p)memory": prefill_summary_dict["memory"],
            "(p)backend": prefill_summary_dict["backend"],
            "(p)version": prefill_summary_dict["version"],
            "(p)system": prefill_summary_dict["system"],
            "(d)tp": decode_summary_dict["tp"],
            "(d)pp": decode_summary_dict["pp"],
            "(d)dp": decode_summary_dict["dp"],
            "(d)moe_tp": decode_summary_dict["moe_tp"],
            "(d)moe_ep": decode_summary_dict["moe_ep"],
            "(d)parallel": decode_summary_dict["parallel"],
            "(d)gemm": decode_summary_dict["gemm"],
            "(d)kvcache": decode_summary_dict["kvcache"],
            "(d)fmha": decode_summary_dict["fmha"],
            "(d)moe": decode_summary_dict["moe"],
            "(d)comm": decode_summary_dict["comm"],
            "(d)memory": decode_summary_dict["memory"],
            "(d)backend": decode_summary_dict["backend"],
            "(d)version": decode_summary_dict["version"],
            "(d)system": decode_summary_dict["system"],
            "power_w": disagg_power_avg,  # Weighted average power for DISAGG mode
            "kv_network_latency_ms": 0.0,  # KV Cache Transfer Latency
            "kv_cache_size_bytes": 0,  # Populated later by caller with KV cache size
        }

    def _get_disagg_summary_df(
        self,
        prefill_summary_df: pd.DataFrame,
        prefill_num_worker: int,
        decode_summary_df: pd.DataFrame,
        decode_num_worker: int,
    ) -> pd.DataFrame:
        """
        Get the disagg summary df based on prefill and decode summary df
        """
        prefill_dict = prefill_summary_df.iloc[0].to_dict()
        decode_dict = decode_summary_df.iloc[0].to_dict()

        summary_dict = self._get_disagg_summary_dict(prefill_dict, prefill_num_worker, decode_dict, decode_num_worker)
        return pd.DataFrame([summary_dict], columns=common.ColumnsDisagg).round(3)

    def _compute_kv_cache_transfer_size(
        self,
        prefill_model: models.BaseModel,
        runtime_config: config.RuntimeConfig,
        prefill_batch_size: int,
    ) -> int:
        """
            Compute the KV cache size to transfer from prefill to decode workers
            using the backend network-transfer KV sizing helper.

            Returns:
                int: KV cache size in bytes
        """
        per_attn_replica_transfer_size = self._prefill_backend.get_total_kv_cache_transfer_size_bytes(
            model=prefill_model,
            batch_size=prefill_batch_size,
            isl=runtime_config.isl,
            beam_width=1,
            osl=0,
        )
        attention_dp_size = max(
            int(getattr(prefill_model.config, "attention_dp_size", 1) or 1),
            1,
        )
        transfer_size = per_attn_replica_transfer_size * attention_dp_size

        logger.debug(
            "KV cache transfer size: %sB per attention replica × dp=%s = %sB total "
            "(batch_size=%s, isl=%s)",
            per_attn_replica_transfer_size,
            attention_dp_size,
            transfer_size,
            prefill_batch_size,
            runtime_config.isl,
        )

        return transfer_size

    def _build_gpu_layout(
        self,
        prefill_model_config: config.ModelConfig,
        prefill_num_worker: int,
        decode_model_config: config.ModelConfig,
        decode_num_worker: int,
    ) -> dict:
        """
        Build GPU layout mapping for prefill-to-decode transfers.
        
        Returns:
            dict: {
                'prefill_workers': flat GPU IDs for prefill workers,
                'decode_workers': flat GPU IDs for decode workers,
                'prefill_worker_layouts': PP/TP-structured prefill worker layouts,
                'decode_worker_layouts': PP/TP-structured decode worker layouts,
                'gpu_layout_strategy': placement strategy used to assign GPU IDs,
                'max_gpu_id_plus_one': highest assigned GPU ID + 1,
            }
        """
        def build_worker_layout(
            worker_id: int,
            tp_size: int,
            pp_size: int,
            start_gpu: int,
            *,
            enable_afd: bool = False,
            attn_tp_size: int = 0,
            attn_dp_size: int = 1,
            ffn_tp_size: int = 0,
            ffn_ep_size: int = 1,
        ) -> dict:
            """Build a single worker's GPU layout.

            Non-AFD (default):
                GPUs are shared between attention and FFN/MoE work.
                Each PP stage has ``attn_tp_size * attn_dp_size`` GPUs.
                ``pp_stages`` is a list of ``pp_size`` lists, each
                containing all stage GPUs.  The per-attention-DP
                grouping is stored in ``attn_dp_pp_stages[dp_rank]``.

            AFD mode (``enable_afd=True``):
                Attention and FFN GPUs are **physically separate groups**.
                Within each PP stage the layout is:

                    [ attn DP-replica 0 (tp GPUs, tp0,tp1...) ]
                    [ attn DP-replica 1 (tp GPUs, tp0,tp1...) ]
                    ...  (attn_dp_size replicas)
                    [ ffn  EP-group 0   (moe_tp GPUs,moe_tp0,moe_tp1...) ]
                    [ ffn  EP-group 1   (moe_tp GPUs,moe_tp0,moe_tp1...) ]
                    ...  (ffn_ep_size groups)

                ``num_attn_gpus_per_stage = attn_tp_size * attn_dp_size``
                ``num_ffn_gpus_per_stage  = ffn_tp_size  * ffn_ep_size``
                ``gpus_per_stage = num_attn + num_ffn``

                ``pp_stages`` is set to the **attention DP-replica-0**
                GPU IDs only (``tp_size`` GPUs per stage) so that the
                existing ``_build_worker_pp_transfer_plan()`` in AstraSim
                works unchanged for the DP-replica-0 transfer path.

                The full per-DP-replica attention layout is stored in
                ``attn_dp_pp_stages[dp_rank]`` — a list of ``pp_size``
                lists, each containing ``attn_tp_size`` GPU IDs.
            """
            if not enable_afd:
                effective_attn_tp = attn_tp_size or tp_size
                effective_attn_dp = max(attn_dp_size, 1)
                gpus_per_stage = effective_attn_tp * effective_attn_dp

                pp_stages = []
                flat_gpu_ids = []
                attn_dp_pp_stages: list[list[list[int]]] = [
                    [] for _ in range(effective_attn_dp)
                ]
                for pp_rank in range(pp_size):
                    stage_start = start_gpu + pp_rank * gpus_per_stage
                    stage_gpu_ids = []
                    for dp_rank in range(effective_attn_dp):
                        replica_start = stage_start + dp_rank * effective_attn_tp
                        replica_gpus = list(
                            range(replica_start, replica_start + effective_attn_tp)
                        )
                        attn_dp_pp_stages[dp_rank].append(replica_gpus)
                        stage_gpu_ids.extend(replica_gpus)
                    pp_stages.append(stage_gpu_ids)
                    flat_gpu_ids.extend(stage_gpu_ids)
                return {
                    "worker_id": worker_id,
                    "gpu_ids": flat_gpu_ids,
                    "pp_stages": pp_stages,
                    "first_stage_gpus": pp_stages[0] if pp_stages else [],
                    "last_stage_gpus": pp_stages[-1] if pp_stages else [],
                    "tp_size": effective_attn_tp,
                    "pp_size": pp_size,
                    "enable_afd": False,
                    "attn_dp_pp_stages": attn_dp_pp_stages,
                    "ffn_pp_stages": None,
                    "attn_gpu_ids": None,
                    "ffn_gpu_ids": None,
                    "attn_tp_size": effective_attn_tp,
                    "attn_dp_size": effective_attn_dp,
                    "ffn_tp_size": ffn_tp_size,
                    "ffn_ep_size": ffn_ep_size,
                }

            # ── AFD path ─────────────────────────────────────────────
            #
            # Per PP stage, GPU layout (contiguous):
            #   attn DP-replica 0: [gpu .. gpu+attn_tp-1]
            #   attn DP-replica 1: [gpu+attn_tp .. gpu+2*attn_tp-1]
            #   ...  (attn_dp_size replicas)
            #   ffn  EP-group 0:   [...]
            #   ffn  EP-group 1:   [...]
            #   ...  (ffn_ep_size groups)
            #
            num_attn_per_stage = attn_tp_size * attn_dp_size
            num_ffn_per_stage = ffn_tp_size * ffn_ep_size
            gpus_per_stage = num_attn_per_stage + num_ffn_per_stage

            all_gpu_ids: list[int] = []
            attn_gpu_ids: list[int] = []
            ffn_gpu_ids: list[int] = []

            # attn_dp_pp_stages[dp_rank][pp_rank] = [gpu_ids for that TP group]
            attn_dp_pp_stages: list[list[list[int]]] = [
                [] for _ in range(attn_dp_size)
            ]
            ffn_pp_stages: list[list[int]] = []  # ffn_pp_stages[pp_rank] = all ffn gpus

            for pp_rank in range(pp_size):
                stage_start = start_gpu + pp_rank * gpus_per_stage

                # Attention sub-group: dp_size replicas of tp_size GPUs
                attn_stage_start = stage_start
                for dp_rank in range(attn_dp_size):
                    replica_start = attn_stage_start + dp_rank * attn_tp_size
                    replica_gpus = list(range(replica_start, replica_start + attn_tp_size))
                    attn_dp_pp_stages[dp_rank].append(replica_gpus)
                    attn_gpu_ids.extend(replica_gpus)
                    all_gpu_ids.extend(replica_gpus)

                # FFN sub-group: ep_size groups of moe_tp GPUs
                ffn_stage_start = stage_start + num_attn_per_stage
                ffn_stage_gpus: list[int] = []
                for ep_rank in range(ffn_ep_size):
                    group_start = ffn_stage_start + ep_rank * ffn_tp_size
                    group_gpus = list(range(group_start, group_start + ffn_tp_size))
                    ffn_stage_gpus.extend(group_gpus)
                ffn_pp_stages.append(ffn_stage_gpus)
                ffn_gpu_ids.extend(ffn_stage_gpus)
                all_gpu_ids.extend(ffn_stage_gpus)

            # pp_stages for KV transfer = DP-replica-0 attention layout.
            # This makes the existing _build_worker_pp_transfer_plan() work
            # unchanged for the common dp=1 case.  When dp>1, the caller
            # (build_kv_transfer_plan) iterates attn_dp_pp_stages.
            pp_stages = attn_dp_pp_stages[0]

            return {
                "worker_id": worker_id,
                "gpu_ids": all_gpu_ids,
                "pp_stages": pp_stages,
                "first_stage_gpus": pp_stages[0] if pp_stages else [],
                "last_stage_gpus": pp_stages[-1] if pp_stages else [],
                "tp_size": attn_tp_size,
                "pp_size": pp_size,
                # ── AFD-specific fields ──────────────────────────────
                "enable_afd": True,
                "attn_dp_pp_stages": attn_dp_pp_stages,
                "ffn_pp_stages": ffn_pp_stages,
                "attn_gpu_ids": attn_gpu_ids,
                "ffn_gpu_ids": ffn_gpu_ids,
                "attn_tp_size": attn_tp_size,
                "attn_dp_size": attn_dp_size,
                "ffn_tp_size": ffn_tp_size,
                "ffn_ep_size": ffn_ep_size,
            }

        ## Only useful when you align the number of gpus up to node boundaries for faster communication.
        def align_up(value: int, alignment: int) -> int:
            if alignment <= 0:
                return value
            return ((value + alignment - 1) // alignment) * alignment

        prefill_tp = prefill_model_config.tp_size
        prefill_pp = prefill_model_config.pp_size
        prefill_afd = prefill_model_config.enable_afd
        p_attn_tp = prefill_tp
        p_attn_dp = max(prefill_model_config.attention_dp_size, 1)
        if prefill_afd:
            p_ffn_tp = prefill_model_config.moe_tp_size or prefill_tp
            p_ffn_ep = prefill_model_config.moe_ep_size or 1
            prefill_gpus_per_worker = (p_attn_tp * p_attn_dp + p_ffn_tp * p_ffn_ep) * prefill_pp
        else:
            if prefill_model_config.moe_tp_size is not None or prefill_model_config.moe_ep_size is not None:
                p_ffn_tp = prefill_model_config.moe_tp_size or prefill_tp
                p_ffn_ep = prefill_model_config.moe_ep_size or 1
            else:
                p_ffn_tp = 0
                p_ffn_ep = 0
            prefill_gpus_per_worker = p_attn_tp * p_attn_dp * prefill_pp

        decode_tp = decode_model_config.tp_size
        decode_pp = decode_model_config.pp_size
        decode_afd = decode_model_config.enable_afd
        d_attn_tp = decode_tp
        d_attn_dp = max(decode_model_config.attention_dp_size, 1)
        if decode_afd:
            d_ffn_tp = decode_model_config.moe_tp_size or decode_tp
            d_ffn_ep = decode_model_config.moe_ep_size or 1
            decode_gpus_per_worker = (d_attn_tp * d_attn_dp + d_ffn_tp * d_ffn_ep) * decode_pp
        else:
            if decode_model_config.moe_tp_size is not None or decode_model_config.moe_ep_size is not None:
                d_ffn_tp = decode_model_config.moe_tp_size or decode_tp
                d_ffn_ep = decode_model_config.moe_ep_size or 1
            else:
                d_ffn_tp = 0
                d_ffn_ep = 0
            decode_gpus_per_worker = d_attn_tp * d_attn_dp * decode_pp

        prefill_worker_layouts = []
        decode_worker_layouts = []
        
        
        def _build_prefill_decode_pairing(
            prefill_num_worker: int, decode_num_worker: int
        ) -> dict[int, list[int]]:
            """
            Build prefill→decode pairing using a contiguous boundary-split
            so load differs by at most one unit.

            P >= D:  each prefill maps to a single decode (single-element list).
                     Example — P=10, D=4:
                       P0→[D0] P1→[D0] P2→[D0] P3→[D1] P4→[D1] P5→[D1]
                       P6→[D2] P7→[D2] P8→[D3] P9→[D3]

            P <  D:  each prefill fans out to a contiguous group of decodes.
                     Example — P=4, D=10:
                       P0→[D0,D1,D2]  P1→[D3,D4,D5]  P2→[D6,D7]  P3→[D8,D9]

            Returns:
                prefill_decode_pairing  dict[p_idx, list[d_idx]]
            """
            prefill_decode_pairing: dict[int, list[int]] = {}
            if prefill_num_worker >= decode_num_worker and decode_num_worker > 0:
                # P >= D: each prefill maps to one decode
                base = prefill_num_worker // decode_num_worker
                base_upper = math.ceil(prefill_num_worker / decode_num_worker)
                remainder = prefill_num_worker % decode_num_worker
                boundary = remainder * base_upper
                for p_idx in range(prefill_num_worker):
                    if p_idx < boundary:
                        d_idx = p_idx // base_upper
                    else:
                        d_idx = remainder + (p_idx - boundary) // base if base > 0 else remainder
                    prefill_decode_pairing[p_idx] = [d_idx]
            else:
                # P < D: each prefill fans out to a contiguous group of decodes
                base = decode_num_worker // prefill_num_worker if prefill_num_worker > 0 else 1
                base_upper = math.ceil(decode_num_worker / prefill_num_worker) if prefill_num_worker > 0 \
                    else decode_num_worker
                remainder = decode_num_worker % prefill_num_worker if prefill_num_worker > 0 else 0
                boundary = remainder * base_upper
                for d_idx in range(decode_num_worker):
                    if d_idx < boundary:
                        p_idx = d_idx // base_upper
                    else:
                        p_idx = remainder + (d_idx - boundary) // base if base > 0 else remainder
                    prefill_decode_pairing.setdefault(p_idx, []).append(d_idx)

            return prefill_decode_pairing

        # Build pairing once — used by both layout strategies and the return dict
        prefill_decode_pairing = _build_prefill_decode_pairing(
            prefill_num_worker, decode_num_worker
        )

        if self._gpu_layout_strategy == "segregated_by_phase":
            # All prefill GPUs packed first, then all decode GPUs.
            # No co-location awareness — simplest layout.
            for worker_id in range(prefill_num_worker):
                worker_start = worker_id * prefill_gpus_per_worker
                prefill_worker_layouts.append(
                    build_worker_layout(
                        worker_id, prefill_tp, prefill_pp, worker_start,
                        enable_afd=prefill_afd,
                        attn_tp_size=p_attn_tp, attn_dp_size=p_attn_dp,
                        ffn_tp_size=p_ffn_tp, ffn_ep_size=p_ffn_ep,
                    )
                )
            decode_gpu_start = prefill_num_worker * prefill_gpus_per_worker
            for worker_id in range(decode_num_worker):
                worker_start = decode_gpu_start + worker_id * decode_gpus_per_worker
                decode_worker_layouts.append(
                    build_worker_layout(
                        worker_id, decode_tp, decode_pp, worker_start,
                        enable_afd=decode_afd,
                        attn_tp_size=d_attn_tp, attn_dp_size=d_attn_dp,
                        ffn_tp_size=d_ffn_tp, ffn_ep_size=d_ffn_ep,
                    )
                )
        else:
            # ── paired_prefill_decode_per_node ───────────────────────
            # Pack transfer groups contiguously so that prefills and
            # their decode targets share the same physical node(s).
            #
            # P >= D:  group by decode worker → [prefills…, decode].
            #          The decode is placed at the END so that the
            #          nearest prefills get NVLink.
            #   Example: 4P:2D, TP=2, 8 GPUs/node → 12 GPUs total
            #     Node 0: P0:{0,1} P1:{2,3} D0:{4,5} P2:{6,7}
            #     Node 1: P3:{8,9} D1:{10,11}
            #
            # P <  D:  group by prefill worker → [prefill, decodes…].
            #          Each prefill is co-located with its decode targets.
            #   Example: 2P:4D, TP=2 → 12 GPUs total
            #     P0:{0,1} D0:{2,3} D1:{4,5} P1:{6,7} D2:{8,9} D3:{10,11}

            # Build contiguous groups: list of (prefill_indices, decode_indices)
            groups: list[tuple[list[int], list[int]]] = []
            if prefill_num_worker >= decode_num_worker:
                # Group by decode worker — invert prefill_decode_pairing on-the-fly
                decode_to_prefills: dict[int, list[int]] = {}
                for p_idx, d_list in prefill_decode_pairing.items():
                    decode_to_prefills.setdefault(d_list[0], []).append(p_idx)
                for d_idx in sorted(decode_to_prefills.keys()):
                    groups.append((decode_to_prefills[d_idx], [d_idx]))
            else:
                # Group by prefill worker
                for p_idx in sorted(prefill_decode_pairing.keys()):
                    groups.append(([p_idx], prefill_decode_pairing[p_idx]))

            cursor = 0
            for group_prefills, group_decodes in groups:
                for p_idx in group_prefills:
                    prefill_worker_layouts.append(
                        build_worker_layout(
                            p_idx, prefill_tp, prefill_pp, cursor,
                            enable_afd=prefill_afd,
                            attn_tp_size=p_attn_tp, attn_dp_size=p_attn_dp,
                            ffn_tp_size=p_ffn_tp, ffn_ep_size=p_ffn_ep,
                        )
                    )
                    cursor += prefill_gpus_per_worker
                for d_idx in group_decodes:
                    decode_worker_layouts.append(
                        build_worker_layout(
                            d_idx, decode_tp, decode_pp, cursor,
                            enable_afd=decode_afd,
                            attn_tp_size=d_attn_tp, attn_dp_size=d_attn_dp,
                            ffn_tp_size=d_ffn_tp, ffn_ep_size=d_ffn_ep,
                        )
                    )
                    cursor += decode_gpus_per_worker

            # Sort layouts by worker_id so indexing matches pairing.
            prefill_worker_layouts.sort(key=lambda w: w["worker_id"])
            decode_worker_layouts.sort(key=lambda w: w["worker_id"])

        used_gpu_ids = [
            gpu_id
            for worker in prefill_worker_layouts + decode_worker_layouts
            for gpu_id in worker["gpu_ids"]
        ]

        return {
            'prefill_workers': [worker["gpu_ids"] for worker in prefill_worker_layouts],
            'decode_workers': [worker["gpu_ids"] for worker in decode_worker_layouts],
            'prefill_worker_layouts': prefill_worker_layouts,
            'decode_worker_layouts': decode_worker_layouts,
            'prefill_decode_pairing': prefill_decode_pairing,
            'prefill_gpus_per_worker': prefill_gpus_per_worker,
            'decode_gpus_per_worker': decode_gpus_per_worker,
            'total_prefill_gpus': prefill_num_worker * prefill_gpus_per_worker,
            'total_decode_gpus': decode_num_worker * decode_gpus_per_worker,
            'gpu_layout_strategy': self._gpu_layout_strategy,
            'max_gpu_id_plus_one': (max(used_gpu_ids) + 1) if used_gpu_ids else 0,
        }

    def _build_model_config_from_summary_row(
        self,
        base_model_config: config.ModelConfig,
        summary_row: pd.Series,
        prefix: str,
    ) -> config.ModelConfig:
        """
        Reconstruct a model config from a summary row so row-specific network
        estimates use the same TP/PP/DP/MoE config as the candidate.
        """
        model_config = copy.deepcopy(base_model_config)
        model_config.tp_size = int(summary_row[f"({prefix})tp"])
        model_config.pp_size = int(summary_row[f"({prefix})pp"])
        model_config.attention_dp_size = int(summary_row[f"({prefix})dp"])
        model_config.moe_tp_size = int(summary_row[f"({prefix})moe_tp"])
        model_config.moe_ep_size = int(summary_row[f"({prefix})moe_ep"])
        return model_config

    def _simulate_network_transfer(
        self,
        gpu_layout: dict,
        kv_cache_size: int,
        prefill_batch_size: int,
    ) -> float:
        """
        Simulate network transfer latency for KV cache from prefill to decode
        using the AstraSimManager.
        
        Args:
            gpu_layout: GPU assignment from _build_gpu_layout
            kv_cache_size: Total KV cache size in bytes
            prefill_batch_size: Number of sequences in the batch
            
        Returns:
            float: Network transfer latency in milliseconds
        """
        return self._astrasim.simulate_kv_cache_transfer(
            gpu_layout=gpu_layout,
            kv_cache_size=kv_cache_size,
            prefill_batch_size=prefill_batch_size,
        )

    def run_disagg(
        self,
        model_path: str,
        runtime_config: config.RuntimeConfig,
        prefill_model_config: config.ModelConfig,
        prefill_batch_size: int,
        prefill_num_worker: int,
        decode_model_config: config.ModelConfig,
        decode_batch_size: int,
        decode_num_worker: int,
    ) -> InferenceSummary:
        """
        Run disagg with given prefill/decode worker info

        Args:
            model_path (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_batch_size (int): the prefill batch size
            prefill_num_worker (int): the number of prefill workers
            decode_model_config (ModelConfig): the decode model config
            decode_batch_size (int): the decode batch size
            decode_num_worker (int): the number of decode workers

        Returns:
            InferenceSummary: the summary of the inference result
        """
        prefill_model = models.get_model(model_path, prefill_model_config, self._prefill_backend.name.value)
        decode_model = models.get_model(model_path, decode_model_config, self._decode_backend.name.value)
        prefill_sess = InferenceSession(
            model=prefill_model, database=self._prefill_database, backend=self._prefill_backend,
            afd_config=self._prefill_afd_config,
        )
        decode_sess = InferenceSession(
            model=decode_model, database=self._decode_database, backend=self._decode_backend,
            afd_config=self._decode_afd_config,
        )

        prefill_runtime_config = copy.deepcopy(runtime_config)
        prefill_runtime_config.batch_size = prefill_batch_size
        prefill_summary = prefill_sess.run_static(mode="static_ctx", runtime_config=prefill_runtime_config)
        
        decode_runtime_config = copy.deepcopy(runtime_config)
        decode_runtime_config.batch_size = decode_batch_size
        decode_summary = decode_sess.run_static(mode="static_gen", runtime_config=decode_runtime_config)

        # === NEW: Network simulation for KV cache transfer ===
        # 1. Compute KV cache size
        kv_cache_size = self._compute_kv_cache_transfer_size(
            prefill_model=prefill_model,
            runtime_config=runtime_config,
            prefill_batch_size=prefill_batch_size,
        )
        logger.info(f"KV cache transfer size: {kv_cache_size / 1e6:.2f} MB")

        # 2. Build GPU layout
        gpu_layout = self._build_gpu_layout(
            prefill_model_config=prefill_model_config,
            prefill_num_worker=prefill_num_worker,
            decode_model_config=decode_model_config,
            decode_num_worker=decode_num_worker,
        )
        logger.info(f"GPU layout: prefill={gpu_layout['prefill_workers']}, decode={gpu_layout['decode_workers']}")

        # 3. Simulate network transfer
        
        kv_network_latency_ms = self._simulate_network_transfer(
            gpu_layout=gpu_layout,
            kv_cache_size=kv_cache_size,
            prefill_batch_size=prefill_batch_size,
        )
        
        if not self._enable_astrasim:
            kv_network_latency_ms = 0.0  # AstraSim not available; skip network latency
        else:
            logger.info(f"Network transfer latency: {kv_network_latency_ms:.3f} ms")
        # === END: Network simulation ===

        disagg_summary_df = self._get_disagg_summary_df(
            prefill_summary.get_summary_df(),
            prefill_num_worker,
            decode_summary.get_summary_df(),
            decode_num_worker,
        )

        # Store network transfer info in the summary DataFrame
        disagg_summary_df['kv_network_latency_ms'] = kv_network_latency_ms
        disagg_summary_df['kv_cache_size_bytes'] = kv_cache_size

        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary.set_summary_df(disagg_summary_df)
        disagg_summary.set_network_info(
            kv_cache_size_bytes=kv_cache_size,
            kv_network_latency_ms=kv_network_latency_ms,
            gpu_layout=gpu_layout,
        )
        
        return disagg_summary

    # ---- AFD (Attention-FFN Disaggregation) for disaggregated inference ----

    def run_disagg_afd(
        self,
        model_path: str,
        runtime_config: config.RuntimeConfig,
        prefill_model_config: config.ModelConfig,
        prefill_batch_size: int,
        prefill_num_worker: int,
        decode_model_config: config.ModelConfig,
        decode_batch_size: int,
        decode_num_worker: int,
    ) -> dict:
        """
        Run disaggregated inference with AFD (Attention-FFN Disaggregation) breakdown.

        This method performs the same work as ``run_disagg`` but additionally
        runs every AFD sub-mode so that callers can inspect how much of each
        phase (prefill / decode) is spent on attention vs FFN operations.

        Args:
            model_path: HuggingFace model path.
            runtime_config: Runtime configuration (isl, osl, batch_size, …).
            prefill_model_config: Model parallelism config for prefill workers.
            prefill_batch_size: Batch size for each prefill worker.
            prefill_num_worker: Number of prefill worker replicas.
            decode_model_config: Model parallelism config for decode workers.
            decode_batch_size: Batch size for each decode worker.
            decode_num_worker: Number of decode worker worker replicas.

        Returns:
            dict with the following keys:

            * ``"disagg_summary"`` – the full ``InferenceSummary`` (same as
              ``run_disagg`` would return).
            * ``"prefill_full"`` – ``InferenceSummary`` for ``static_ctx``.
            * ``"prefill_attn"`` – ``InferenceSummary`` for ``static_ctx_attn``.
            * ``"prefill_ffn"``  – ``InferenceSummary`` for ``static_ctx_ffn``.
            * ``"decode_full"``  – ``InferenceSummary`` for ``static_gen``.
            * ``"decode_attn"``  – ``InferenceSummary`` for ``static_gen_attn``.
            * ``"decode_ffn"``   – ``InferenceSummary`` for ``static_gen_ffn``.
            * ``"prefill_attn_pct"`` – attention share of prefill latency (%).
            * ``"prefill_ffn_pct"``  – FFN share of prefill latency (%).
            * ``"decode_attn_pct"``  – attention share of decode latency (%).
            * ``"decode_ffn_pct"``   – FFN share of decode latency (%).
        """
        # Build models & sessions (same as run_disagg)
        prefill_model = models.get_model(
            model_path, prefill_model_config, self._prefill_backend.name.value
        )
        decode_model = models.get_model(
            model_path, decode_model_config, self._decode_backend.name.value
        )
        prefill_sess = InferenceSession(
            model=prefill_model,
            database=self._prefill_database,
            backend=self._prefill_backend,
            afd_config=self._prefill_afd_config,
        )
        decode_sess = InferenceSession(
            model=decode_model,
            database=self._decode_database,
            backend=self._decode_backend,
            afd_config=self._decode_afd_config,
        )

        # Prepare per-phase runtime configs
        prefill_runtime_config = copy.deepcopy(runtime_config)
        prefill_runtime_config.batch_size = prefill_batch_size

        decode_runtime_config = copy.deepcopy(runtime_config)
        decode_runtime_config.batch_size = decode_batch_size

        # --- Run all six sub-modes ---
        prefill_full = prefill_sess.run_static(
            runtime_config=prefill_runtime_config, mode="static_ctx"
        )
        prefill_attn = prefill_sess.run_static(
            runtime_config=prefill_runtime_config, mode="static_ctx_attn"
        )
        prefill_ffn = prefill_sess.run_static(
            runtime_config=prefill_runtime_config, mode="static_ctx_ffn"
        )

        decode_full = decode_sess.run_static(
            runtime_config=decode_runtime_config, mode="static_gen"
        )
        decode_attn = decode_sess.run_static(
            runtime_config=decode_runtime_config, mode="static_gen_attn"
        )
        decode_ffn = decode_sess.run_static(
            runtime_config=decode_runtime_config, mode="static_gen_ffn"
        )

        # --- Build the combined disagg summary (same as run_disagg) ---
        # === Network simulation for KV cache transfer (same as run_disagg) ===
        kv_cache_size = self._compute_kv_cache_transfer_size(
            prefill_model=prefill_model,
            runtime_config=runtime_config,
            prefill_batch_size=prefill_batch_size,
        )
        gpu_layout = self._build_gpu_layout(
            prefill_model_config=prefill_model_config,
            prefill_num_worker=prefill_num_worker,
            decode_model_config=decode_model_config,
            decode_num_worker=decode_num_worker,
        )
        kv_network_latency_ms = self._simulate_network_transfer(
            gpu_layout=gpu_layout,
            kv_cache_size=kv_cache_size,
            prefill_batch_size=prefill_batch_size,
        )
        if not self._enable_astrasim:
            kv_network_latency_ms = 0.0
        else:
            logger.info(f"AFD network transfer latency: {kv_network_latency_ms:.3f} ms")
        # === END: Network simulation ===

        disagg_summary_df = self._get_disagg_summary_df(
            prefill_full.get_summary_df(),
            prefill_num_worker,
            decode_full.get_summary_df(),
            decode_num_worker,
        )

        # Store network transfer info in the summary DataFrame
        disagg_summary_df['kv_network_latency_ms'] = kv_network_latency_ms
        disagg_summary_df['kv_cache_size_bytes'] = kv_cache_size

        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary.set_summary_df(disagg_summary_df)
        disagg_summary.set_network_info(
            kv_cache_size_bytes=kv_cache_size,
            kv_network_latency_ms=kv_network_latency_ms,
            gpu_layout=gpu_layout,
        )

        # --- Compute percentage breakdowns ---
        prefill_total = sum(prefill_full.get_context_latency_dict().values())
        prefill_attn_total = sum(prefill_attn.get_context_latency_dict().values())
        prefill_ffn_total = sum(prefill_ffn.get_context_latency_dict().values())

        decode_total = sum(decode_full.get_generation_latency_dict().values())
        decode_attn_total = sum(decode_attn.get_generation_latency_dict().values())
        decode_ffn_total = sum(decode_ffn.get_generation_latency_dict().values())

        prefill_attn_pct = (
            prefill_attn_total / prefill_total * 100.0 if prefill_total > 0 else 0.0
        )
        prefill_ffn_pct = (
            prefill_ffn_total / prefill_total * 100.0 if prefill_total > 0 else 0.0
        )
        decode_attn_pct = (
            decode_attn_total / decode_total * 100.0 if decode_total > 0 else 0.0
        )
        decode_ffn_pct = (
            decode_ffn_total / decode_total * 100.0 if decode_total > 0 else 0.0
        )

        logger.info(
            f"AFD Breakdown — Prefill: attn={prefill_attn_pct:.1f}% ffn={prefill_ffn_pct:.1f}%  "
            f"Decode: attn={decode_attn_pct:.1f}% ffn={decode_ffn_pct:.1f}%"
        )

        return {
            "disagg_summary": disagg_summary,
            "prefill_full": prefill_full,
            "prefill_attn": prefill_attn,
            "prefill_ffn": prefill_ffn,
            "decode_full": decode_full,
            "decode_attn": decode_attn,
            "decode_ffn": decode_ffn,
            "prefill_attn_pct": prefill_attn_pct,
            "prefill_ffn_pct": prefill_ffn_pct,
            "decode_attn_pct": decode_attn_pct,
            "decode_ffn_pct": decode_ffn_pct,
        }

    # optimization
    def find_best_disagg_result_under_constraints(
        self,
        model_path: str,
        runtime_config: config.RuntimeConfig,
        prefill_model_config: config.ModelConfig,
        prefill_parallel_config_list: list[tuple[int, int, int, int, int]],
        prefill_max_num_tokens: int,
        prefill_num_worker_list: list[int],
        decode_model_config: config.ModelConfig,
        decode_parallel_config_list: list[tuple[int, int, int, int, int]],
        decode_max_num_tokens: int,
        decode_num_worker_list: list[int],
        num_gpu_list: list[int] | None,
    ) -> InferenceSummary | None:
        """
        Run disagg with given constraints
        1. get all summary df, which matches the constraints
        2. find best config under constraints, call match scales to get the best scale
        3. call a func to get disagg_summary_df (this is shared by run_disgg func)
        4. return summary
        5. several empirical values:
            - 0.7 is the threshold to filter decode workers, because the performance of
              decode workers is much lower than prefill workers
            - 5 is the top k to return for drawing pareto frontier of each tpot

        Args:
            model_path (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_parallel_config_list (List[Tuple[int, int, int, int, int]]):
                the prefill parallel config list
            prefill_max_num_tokens (int): the prefill max num tokens
            prefill_num_worker_list (List[int]): the prefill num worker list
            decode_model_config (ModelConfig): the decode model config
            decode_parallel_config_list (List[Tuple[int, int, int, int, int]]):
                the decode parallel config list
            decode_max_num_tokens (int): the decode max num tokens
            decode_num_worker_list (List[int]): the decode num worker list
            num_gpu_list (Optional[List[int]]): the num gpu list

        Returns:
            Optional[InferenceSummary]: the summary of the inference result, contains all the
                possible disagg config and perf that matches SLA.
        """

        # minor perf optimization: convert num_gpu_list to a set to speed up lookup
        num_gpu_set = set[int](num_gpu_list) if num_gpu_list else set()

        @functools.lru_cache(maxsize=8192)
        def _match_workers(
            prefill_throughput: float,
            prefill_gpus: int,
            decode_throughput: float,
            decode_gpus: int,
            rate_matching_prefill_degradation_factor: float,
            rate_matching_decode_degradation_factor: float,
        ) -> tuple[int, int]:
            """
            Match the prefill and decode workers, return the best prefill and decode num worker
            """
            prefill_opt_num_worker, decode_opt_num_worker = -1, -1
            throughput_per_gpu_max = 0
            for decode_num_worker in decode_num_worker_list:
                for prefill_num_worker in prefill_num_worker_list:
                    num_gpu = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker

                    # if num_gpu_set is empty, we don't have any constraint on the number of gpus
                    # if num_gpu_set is not empty, we only consider the gpus that are in the set
                    if len(num_gpu_set) > 0 and num_gpu not in num_gpu_set:
                        continue

                    prefill_throughput_corrected = (
                        prefill_throughput * prefill_num_worker * rate_matching_prefill_degradation_factor
                    )
                    decode_throughput_corrected = (
                        decode_throughput * decode_num_worker * rate_matching_decode_degradation_factor
                    )

                    # criteria 1, try to make prefill_throughput larger than decode_throughput
                    # otherwise, decode bs cannot be achieved and decode throughput cannot be
                    # achieved as well.
                    # if prefill_throughput < decode_throughput:
                    #    continue

                    # criteria 2, try to make the throughput per gpu larger
                    throughput_per_gpu = min(prefill_throughput_corrected, decode_throughput_corrected) / num_gpu

                    if throughput_per_gpu > throughput_per_gpu_max:
                        throughput_per_gpu_max = throughput_per_gpu
                        prefill_opt_num_worker, decode_opt_num_worker = (
                            prefill_num_worker,
                            decode_num_worker,
                        )

            return prefill_opt_num_worker, decode_opt_num_worker

        def _get_summary_df(
            model_config: config.ModelConfig,
            parallel_config_list: list[tuple[int, int, int, int, int]],
            b_list: list[int],
            runtime_config: config.RuntimeConfig,
            mode: str,
            latency_correction_scale: float = 1.0,
        ) -> pd.DataFrame:
            """
            Get all worker candidates based on give search space
            """
            summary_df = pd.DataFrame(columns=common.ColumnsStatic)
            exceptions = []

            for parallel_config in parallel_config_list:
                tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parallel_config
                logger.debug(
                    f"Getting candidate workers with parallel config: tp={tp_size}, pp={pp_size}, "
                    f"dp={dp_size}, moe_tp={moe_tp_size}, moe_ep={moe_ep_size}"
                )

                try:
                    overwritten_model_config = copy.deepcopy(model_config)
                    overwritten_model_config.pp_size = pp_size
                    overwritten_model_config.tp_size = tp_size
                    overwritten_model_config.moe_tp_size = moe_tp_size
                    overwritten_model_config.moe_ep_size = moe_ep_size
                    overwritten_model_config.attention_dp_size = dp_size
                    model = models.get_model(
                        model_path=model_path,
                        model_config=overwritten_model_config,
                        backend_name=self._prefill_backend.name.value,
                    )
                    if mode == "static_ctx":
                        sess = InferenceSession(
                            model=model,
                            database=self._prefill_database,
                            backend=self._prefill_backend,
                            afd_config=self._prefill_afd_config,
                        )
                    else:
                        sess = InferenceSession(
                            model=model,
                            database=self._decode_database,
                            backend=self._decode_backend,
                            afd_config=self._decode_afd_config,
                        )

                    for b in b_list:
                        overwritten_runtime_config = copy.deepcopy(runtime_config)
                        overwritten_runtime_config.batch_size = b
                        summary = sess.run_static(
                            mode=mode,
                            runtime_config=overwritten_runtime_config,
                            latency_correction_scale=latency_correction_scale,
                        )
                        if not summary.check_oom():
                            summary_df = pd.concat(
                                [summary_df, summary.get_summary_df()],
                                axis=0,
                                ignore_index=True,
                            )
                        else:  # larger b will always OOM
                            break
                except Exception as e:
                    logger.exception(
                        f"Error getting candidate workers with parallel config: "
                        f"tp={tp_size}, pp={pp_size}, dp={dp_size}, moe_tp={moe_tp_size}, "
                        f"moe_ep={moe_ep_size}; skipping this combination"
                    )
                    exceptions.append(e)
                    continue
            if summary_df.empty:
                if exceptions:
                    raise RuntimeError(
                        f"No results found for any parallel configuration. Showing last exception: {exceptions[-1]}"
                    ) from exceptions[-1]
                raise RuntimeError(
                    "No results found for any parallel configuration. All configurations resulted in OOM."
                )
            return summary_df

        def _find_best_result_under_constraints(
            ttft: float,
            tpot: float,
            prefill_summary_df: pd.DataFrame,
            decode_summary_df: pd.DataFrame,
            return_top_k: int,
            num_gpu_list: list[int] | None,
            rate_matching_prefill_degradation_factor: float,
            rate_matching_decode_degradation_factor: float,
        ) -> InferenceSummary:
            """
            Find the best result under constraints
            """

            # 1. we categorize the decode summary
            #    df into different categories based on parallelism (we can use the parallel key in
            #    the df). do the rate matching and sort the result by category - throughput.
            # 2. for prefill, follow two rules: high throughput, if at same level, choose the one
            #    with small batchsize. add one func for correct ttft (we have some formula,
            #    just leave it blank for now)
            # 3. prefill/decode correction are already applied to workers.
            #    Additional correction can be a degradation factor for the final result during the
            #   rate matching process.
            # 4. rate matching. The prefill throughput should be 1.x larger than the decode
            #    throughput.
            #    "1.x" is an empirical value. Default is 1.1.

            # only ttft will be corrected here, other latency and throughput will not be
            # corrected. concurrency / num_prefill_workers = local_concurrency(lc);
            # N x concurrency requests. formula = (lc * (lc+1) / 2 + lc * (N-1) )/lc/N
            # if we use N=10, it's lc/20+0.95. assume lc can be 15-20, 1.8 is a reasonable
            # correction factor. as we need to get the lc after rate matching, we cannot get the
            # exact value now. Let's make it simple to do pre-correction instead of post-correction.
            correction_factor = 1.8  # let's make it simple for now.
            prefill_candidates = prefill_summary_df.assign(ttft=prefill_summary_df["ttft"] * correction_factor)

            prefill_candidates = prefill_candidates[prefill_candidates["ttft"] < ttft]
            if len(prefill_candidates) == 0:
                logger.warning(f"No prefill worker candidates found for ttft {ttft}ms.")
                return None
            prefill_candidates = (
                prefill_candidates.sort_values(by=["seq/s/gpu", "global_bs"], ascending=[False, True])
                .reset_index(drop=True)
                .head(MAX_NUM_PREFILL_WORKER_CANDIDATES)
            )

            decode_candidates = decode_summary_df[
                (decode_summary_df["tpot"] < tpot * DECODE_FILTER_RATIO_MAX)
                & (decode_summary_df["tpot"] > tpot * DECODE_FILTER_RATIO_MIN)
            ].copy()
            if len(decode_candidates) == 0:
                logger.warning(f"No decode worker candidates found for tpot {tpot}ms.")
                return None

            all_category_results: list[dict] = []
            prefill_candidates_list = prefill_candidates.to_dict("records")

            for parallel_value, parallel_group in decode_candidates.groupby("parallel"):
                parallel_group_sorted = (
                    parallel_group.sort_values(by=["seq/s/gpu"], ascending=[False])
                    .reset_index(drop=True)
                    .head(MAX_NUM_DECODE_WORKER_CANDIDATES)
                )

                decode_workers_list = parallel_group_sorted.to_dict("records")
                category_results: list[dict] = []
                for decode_worker in decode_workers_list:
                    decode_throughput = float(decode_worker["seq/s"])
                    decode_gpus = decode_worker["num_total_gpus"]
                    for prefill_worker in prefill_candidates_list:
                        prefill_throughput = float(prefill_worker["seq/s"])
                        prefill_gpus = prefill_worker["num_total_gpus"]
                        prefill_num_worker, decode_num_worker = _match_workers(
                            prefill_throughput=prefill_throughput,
                            prefill_gpus=prefill_gpus,
                            decode_throughput=decode_throughput,
                            decode_gpus=decode_gpus,
                            rate_matching_prefill_degradation_factor=rate_matching_prefill_degradation_factor,
                            rate_matching_decode_degradation_factor=rate_matching_decode_degradation_factor,
                        )
                        if prefill_num_worker == -1 or decode_num_worker == -1:
                            continue

                        disagg_dict = self._get_disagg_summary_dict(
                            prefill_worker, prefill_num_worker, decode_worker, decode_num_worker
                        )
                        category_results.append(disagg_dict)

                if category_results:
                    # only return the best one for each category
                    best_result = max(category_results, key=lambda x: (x["tokens/s/gpu"], -x["num_total_gpus"]))
                    all_category_results.append(best_result)
                else:
                    logger.debug(f"No matched result for decode parallel {parallel_value}.")

            if not all_category_results:
                logger.debug("No disagg summary found after applying constraints.")
                return None

            disagg_summary_df = pd.DataFrame(all_category_results, columns=common.ColumnsDisagg).round(3)
            disagg_summary_df = (
                disagg_summary_df.sort_values(by=["tokens/s/gpu"], ascending=[False])
                .head(return_top_k)
                .reset_index(drop=True)
            )
            return disagg_summary_df
            # _find_best_result_under_constraints() ends here

        # start, get all possible p/d servers
        if decode_max_num_tokens < 1:
            logger.warning("decode_max_num_tokens is less than 1, set to 1")
            decode_max_num_tokens = 1
        decode_batch_size_list_default = (
            list(range(1, 16, 1)) + list(range(16, 32, 2)) + list(range(32, 128, 4)) + list(range(128, 512, 8)) + [512]
        )
        if decode_max_num_tokens > max(decode_batch_size_list_default):
            decode_batch_size_range = decode_batch_size_list_default + [decode_max_num_tokens]
        else:
            decode_batch_size_range = [i for i in decode_batch_size_list_default if i <= decode_max_num_tokens]

        if prefill_max_num_tokens < runtime_config.isl:
            logger.warning("prefill_max_num_tokens is less than runtime_config.isl, set to runtime_config.isl")
            prefill_max_num_tokens = runtime_config.isl

        max_prefill_batch_size = prefill_max_num_tokens // runtime_config.isl
        prefill_batch_size_range = range(1, max_prefill_batch_size + 1)

        # initialize disagg summary
        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary_df = pd.DataFrame(columns=common.ColumnsDisagg)
        disagg_summary.set_summary_df(disagg_summary_df)

        # find prefill and decode workers
        prefill_summary_df = _get_summary_df(
            prefill_model_config,
            prefill_parallel_config_list,
            prefill_batch_size_range,
            runtime_config,
            "static_ctx",
            latency_correction_scale=self._prefill_latency_correction_scale,
        )
        decode_summary_df = _get_summary_df(
            decode_model_config,
            decode_parallel_config_list,
            decode_batch_size_range,
            runtime_config,
            "static_gen",  
            latency_correction_scale=self._decode_latency_correction_scale,
        )
        if len(prefill_summary_df) == 0 or len(decode_summary_df) == 0:
            logger.debug(f"No prefill or decode workers found for {model_path} with given configs.")
            return disagg_summary

        # find best result under constraints
        constraint_pairs: list[tuple[float, float]] = []
        if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
            constraint_pairs = enumerate_ttft_tpot_constraints(
                runtime_config.osl,
                runtime_config.request_latency,
                runtime_config.ttft,
            )
            if not constraint_pairs:
                logger.debug(
                    "No ttft/tpot constraints derived for request_latency=%s in disagg optimization.",
                    runtime_config.request_latency,
                )
        else:
            tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
            constraint_pairs = [(runtime_config.ttft, tpot) for tpot in tpot_values]

        for ttft_constraint, tpot_constraint in constraint_pairs:
            logger.debug(
                "Finding best result under constraints for ttft=%sms, tpot=%sms...",
                ttft_constraint,
                tpot_constraint,
            )
            filtered_disagg_summary_df = _find_best_result_under_constraints(
                ttft=ttft_constraint,
                tpot=tpot_constraint,
                prefill_summary_df=prefill_summary_df,
                decode_summary_df=decode_summary_df,
                return_top_k=5,
                num_gpu_list=num_gpu_list,
                rate_matching_prefill_degradation_factor=self._RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
                rate_matching_decode_degradation_factor=self._RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
            )
            if filtered_disagg_summary_df is not None:
                disagg_summary_df = pd.concat(
                    [disagg_summary_df, filtered_disagg_summary_df], axis=0, ignore_index=True
                )
        if len(disagg_summary_df) == 0:
            logger.debug(f"No disagg result found for {model_path} with given constraints.")
            return disagg_summary

        disagg_summary_df = disagg_summary_df.drop_duplicates(ignore_index=True)

        # === AstraSim: Annotate each candidate with KV cache transfer latency ===
        # This is a post-processing step: for each candidate configuration, we
        # estimate the network latency for KV cache transfer and store it as a
        # separate metric using the candidate's own parallel config.
        if self._enable_astrasim and len(disagg_summary_df) > 0:
            for idx in disagg_summary_df.index:
                row = disagg_summary_df.loc[idx]
                p_bs = int(row.get("(p)bs", 1))
                p_workers = int(row.get("(p)workers", 1))
                d_workers = int(row.get("(d)workers", 1))

                try:
                    row_prefill_model_config = self._build_model_config_from_summary_row(
                        base_model_config=prefill_model_config,
                        summary_row=row,
                        prefix="p",
                    )
                    row_decode_model_config = self._build_model_config_from_summary_row(
                        base_model_config=decode_model_config,
                        summary_row=row,
                        prefix="d",
                    )
                    # For KV Cache is only generated on the prefill side
                    row_prefill_model = models.get_model(
                        model_path=model_path,
                        model_config=row_prefill_model_config,
                        backend_name=self._prefill_backend.name.value,
                    )
                    kv_cache_size = self._compute_kv_cache_transfer_size(
                        prefill_model=row_prefill_model,
                        runtime_config=runtime_config,
                        prefill_batch_size=p_bs,
                    )
                    gpu_layout = self._build_gpu_layout(
                        prefill_model_config=row_prefill_model_config,
                        prefill_num_worker=p_workers,
                        decode_model_config=row_decode_model_config,
                        decode_num_worker=d_workers,
                    )
                    net_lat = self._simulate_network_transfer(
                        gpu_layout=gpu_layout,
                        kv_cache_size=kv_cache_size,
                        prefill_batch_size=p_bs,
                    )
                    disagg_summary_df.at[idx, "kv_network_latency_ms"] = net_lat
                    disagg_summary_df.at[idx, "kv_cache_size_bytes"] = kv_cache_size
                except Exception as e:
                    logger.debug(f"AstraSim network latency estimation skipped for row {idx}: {e}")

        # set final disagg summary
        disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary
