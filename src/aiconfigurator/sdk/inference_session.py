# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import logging
import os
import sys
import warnings

import pandas as pd

from aiconfigurator.sdk import common, config, models, perf_database
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
# Get the directory of this file to build relative paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

# Default network config path (relative to project root)
DEFAULT_NETWORK_CONFIG = os.path.join(_PROJECT_ROOT, "network_backend", "astra-network-analytical", "input", "Ring.yml")

# Network simulator library path (relative to project root)
_NETWORK_SIM_LIB_PATH = os.path.join(_PROJECT_ROOT, "network_backend", "astra-network-analytical", "lib")

# TODO: Turn this into input later
USE_ASTRA_SIM = True

# Attempt to import AstraSim network simulator
sys.path.append(_NETWORK_SIM_LIB_PATH)
try:
    from simulation_py_congestion_aware import (
        EventQueue,
        Topology,
        Chunk,
        NetworkParser,
        construct_topology,
    )
    NETWORK_SIM_AVAILABLE = True
    print(f"AstraSim Network simulator loaded from {_NETWORK_SIM_LIB_PATH}")
except ImportError:
    NETWORK_SIM_AVAILABLE = False
    print("AstraSim Network simulator not available, skipping network latency modeling")



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

    def __init__(self, model: models.BaseModel, database: perf_database.PerfDatabase, backend: BaseBackend) -> None:
        """
        Initialize the InferenceSession
        """
        self._model = model
        self._database = database
        self._backend = backend

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
            self._model, self._database, runtime_config, mode, stride, latency_correction_scale
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
        return self._backend.run_agg(self._model, self._database, runtime_config, **kwargs)

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
            self._model, self._database, runtime_config, **kwargs
        )


DECODE_FILTER_RATIO_MIN = 0.0
DECODE_FILTER_RATIO_MAX = 1.0
MAX_DECODE_WORKERS_PER_CATEGORY = 16
MAX_PREFILL_WORKERS = 32
MAX_NUM_DECODE_WORKER_CANDIDATES = 64
MAX_NUM_PREFILL_WORKER_CANDIDATES = 32


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
        network_file: str = None,  # NEW: optional network config file
    ) -> None:
        """
        Initialize the DisaggInferenceSession
        """
        self._prefill_database = prefill_database
        self._prefill_backend = prefill_backend
        self._decode_database = decode_database
        self._decode_backend = decode_backend

        # allow user to set correction scales for better alignment with real system
        self._prefill_latency_correction_scale = 1.0
        self._decode_latency_correction_scale = 1.0

        # comes from pipeline bubble, especially when benchmarking with concurrency
        self._RATE_MATCHING_PREFILL_DEGRADATION_FACTOR = 0.9
        # comes from not saturating the batchsize slot of decode worker
        self._RATE_MATCHING_DECODE_DEGRADATION_FACTOR = 0.92

        # AstraSim network simulator setup
        self._network_file = network_file
        self._enable_astrasim = False
        self._network_queue = None
        self._topology = None

        if NETWORK_SIM_AVAILABLE:
            if network_file is None:
                print(DEFAULT_NETWORK_CONFIG)
                self._network_file = str(DEFAULT_NETWORK_CONFIG)
            logger.info(f"Enabling AstraSim Network Engine with config: {network_file}")
            self._enable_astrasim = True
            self._network_queue = EventQueue()
            network_parser = NetworkParser(self._network_file)
            Topology.set_event_queue(self._network_queue)
            self._topology = construct_topology(network_parser)
        else:
            logger.warning("Network file provided but AstraSim simulator not available")

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
        prefill_gpus = prefill_summary_dict["pp"] * prefill_summary_dict["tp"] * prefill_summary_dict["dp"]
        decode_gpus = decode_summary_dict["pp"] * decode_summary_dict["tp"] * decode_summary_dict["dp"]
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
        model_path: str,
        prefill_model_config: config.ModelConfig,
        runtime_config: config.RuntimeConfig,
        prefill_batch_size: int,
    ) -> int:
        """
            Compute the KV cache size to transfer from prefill to decode workers.
            
            Transfer size = num_prefill_tokens * hidden_size * dtype_size
            
            Returns:
                int: KV cache size in bytes
        """
        from aiconfigurator.sdk.utils import get_model_config_from_model_path
        
        model_arch = get_model_config_from_model_path(model_path)
        
        hidden_size = model_arch[5]  # hidden size is the 6th element in model_arch tuple
        num_prefill_tokens = prefill_batch_size * runtime_config.isl
        
        # KV cache quantization (fp8 = 1 byte, fp16 = 2 bytes)
        kvcache_quant = prefill_model_config.kvcache_quant_mode
        if hasattr(kvcache_quant, 'memory'):
            dtype_size = kvcache_quant.memory
        else:
            dtype_size = 2  # Default to fp16
        
        # Transfer size in bytes
        transfer_size = num_prefill_tokens * hidden_size * dtype_size
        
        # Log in MB
        logger.debug(f"KV cache transfer size: {transfer_size}B "
                     f"({num_prefill_tokens} tokens × {hidden_size} hidden_size × {dtype_size} bytes)")
        
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
                'prefill_gpus': list of GPU IDs for prefill workers,
                'decode_gpus': list of GPU IDs for decode workers,
                'transfers': list of (src_gpu, dst_gpu, size) tuples
            }
        """
        prefill_tp = prefill_model_config.tp_size
        prefill_pp = prefill_model_config.pp_size
        prefill_gpus_per_worker = prefill_tp * prefill_pp
        
        decode_tp = decode_model_config.tp_size
        decode_pp = decode_model_config.pp_size
        decode_gpus_per_worker = decode_tp * decode_pp
        
        # Assign GPU IDs
        # Prefill GPUs: 0 to (prefill_num_worker * prefill_gpus_per_worker - 1)
        # Decode GPUs: next available IDs
        prefill_gpu_ids = []
        for worker_id in range(prefill_num_worker):
            start_gpu = worker_id * prefill_gpus_per_worker
            worker_gpus = list(range(start_gpu, start_gpu + prefill_gpus_per_worker))
            prefill_gpu_ids.append(worker_gpus)
        
        decode_gpu_start = prefill_num_worker * prefill_gpus_per_worker
        decode_gpu_ids = []
        for worker_id in range(decode_num_worker):
            start_gpu = decode_gpu_start + worker_id * decode_gpus_per_worker
            worker_gpus = list(range(start_gpu, start_gpu + decode_gpus_per_worker))
            decode_gpu_ids.append(worker_gpus)
        
        return {
            'prefill_workers': prefill_gpu_ids,  # [[0], [1], [2], [3]] for 4 workers with TP=1
            'decode_workers': decode_gpu_ids,    # [[4, 5], [6, 7]] for 2 workers with TP=2
            'prefill_gpus_per_worker': prefill_gpus_per_worker,
            'decode_gpus_per_worker': decode_gpus_per_worker,
            'total_prefill_gpus': prefill_num_worker * prefill_gpus_per_worker,
            'total_decode_gpus': decode_num_worker * decode_gpus_per_worker,
        }

    def _simulate_network_transfer(
        self,
        gpu_layout: dict,
        kv_cache_size: int,
        prefill_batch_size: int,
    ) -> float:
        """
        Simulate network transfer latency for KV cache from prefill to decode
        using AstraSim network simulator.
        
        Args:
            gpu_layout: GPU assignment from _build_gpu_layout
            kv_cache_size: Total KV cache size in bytes
            prefill_batch_size: Number of sequences in the batch
            
        Returns:
            float: Network transfer latency in milliseconds
        """
        if not self._enable_astrasim:
            logger.debug("AstraSim network simulator not enabled, returning 0 latency")
            return 0.0
        
        logger.info("Using AstraSim Network Engine for KV cache transfer simulation")
        
        # Build transfer schedule
        prefill_workers = gpu_layout['prefill_workers']
        decode_workers = gpu_layout['decode_workers']
        num_prefill_workers = len(prefill_workers)
        num_decode_workers = len(decode_workers)
        
        # KV cache size per sequence
        kv_size_per_seq = kv_cache_size // prefill_batch_size if prefill_batch_size > 0 else kv_cache_size
        
        # Sequences distributed evenly across prefill workers
        seqs_per_prefill = max(1, prefill_batch_size // num_prefill_workers)
        
        try:
            # Reset network queue for fresh simulation
            # self._network_queue = EventQueue()
            # Topology.set_event_queue(self._network_queue)
            # network_parser = NetworkParser(self._network_file)
            # self._topology = construct_topology(network_parser)
            
            # Create and send chunks for each KV cache transfer
            chunk_id = 0
            for p_idx, p_gpus in enumerate(prefill_workers):
                # Last GPU in prefill worker (after PP) sends KV cache
                src_gpu = p_gpus[-1]
                
                for seq_idx in range(seqs_per_prefill):
                    # Round-robin assignment to decode workers
                    d_idx = (p_idx * seqs_per_prefill + seq_idx) % num_decode_workers
                    # First GPU in decode worker receives KV cache
                    dst_gpu = decode_workers[d_idx][0]
                    
                    # Create chunk using AstraSim API
                    # data_size is in bytes
                    chunk = Chunk.create_with_event_queue(
                        kv_size_per_seq,  # data_size in bytes
                        src_gpu,          # source GPU ID
                        dst_gpu,          # destination GPU ID
                        chunk_id,         # request_id / chunk_id
                        self._topology,
                        self._network_queue,
                    )
                    
                    # Send the chunk through the network
                    self._topology.send_python(chunk)
                    
                    logger.info(
                        f"KV transfer chunk {chunk_id}: GPU {src_gpu} -> GPU {dst_gpu}, "
                        f"{kv_size_per_seq} bytes"
                    )
                    chunk_id += 1
            
            # chunk_id = 1  # Placeholder chunk ID
            # chunk = Chunk.create_with_event_queue(
            #     kv_size_per_seq,  # data_size in bytes
            #     0,          # source GPU ID
            #     5,          # destination GPU ID
            #     chunk_id,         # request_id / chunk_id
            #     self._topology,
            #     self._network_queue,
            # )
            # self._topology.send_python(chunk)
            
            # Run simulation until all transfers complete
            # Process all events in the network queue
            while not self._network_queue.finished():
                self._network_queue.proceed()
            # Get the final simulation time (in nanoseconds)
            final_time_ns = self._network_queue.get_current_time()
            
            # Convert nanoseconds to milliseconds
            latency_ms = final_time_ns / 1e6
            
            logger.info(
                f"AstraSim network transfer latency: {latency_ms:.3f} ms "
                f"for {chunk_id} transfers, total {kv_cache_size} bytes"
            )
            return latency_ms
            
        except Exception as e:
            logger.warning(f"AstraSim network simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

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
            model=prefill_model, database=self._prefill_database, backend=self._prefill_backend
        )
        decode_sess = InferenceSession(model=decode_model, database=self._decode_database, backend=self._decode_backend)

        prefill_runtime_config = copy.deepcopy(runtime_config)
        prefill_runtime_config.batch_size = prefill_batch_size
        prefill_summary = prefill_sess.run_static(mode="static_ctx", runtime_config=prefill_runtime_config)
        
        decode_runtime_config = copy.deepcopy(runtime_config)
        decode_runtime_config.batch_size = decode_batch_size
        decode_summary = decode_sess.run_static(mode="static_gen", runtime_config=decode_runtime_config)

        # # === NEW: Network simulation for KV cache transfer ===
        # # 1. Compute KV cache size
        kv_cache_size = self._compute_kv_cache_transfer_size(
            model_path=model_path,
            prefill_model_config=prefill_model_config,
            runtime_config=runtime_config,
            prefill_batch_size=prefill_batch_size,
        )
        logger.info(f"KV cache transfer size: {kv_cache_size / 1e6:.2f} MB")

        # # 2. Build GPU layout
        gpu_layout = self._build_gpu_layout(
            prefill_model_config=prefill_model_config,
            prefill_num_worker=prefill_num_worker,
            decode_model_config=decode_model_config,
            decode_num_worker=decode_num_worker,
        )
        logger.info(f"GPU layout: prefill={gpu_layout['prefill_workers']}, decode={gpu_layout['decode_workers']}")

        # # 3. Simulate network transfer
        network_latency_ms = self._simulate_network_transfer(
            gpu_layout=gpu_layout,
            kv_cache_size=kv_cache_size,
            prefill_batch_size=prefill_batch_size,
        )
        network_latency_ms = 0.0  # Placeholder if network sim is disabled
        # logger.info(f"Network transfer latency: {network_latency_ms:.3f} ms")
        # # === END: Network simulation ===

        disagg_summary_df = self._get_disagg_summary_df(
            prefill_summary.get_summary_df(),
            prefill_num_worker,
            decode_summary.get_summary_df(),
            decode_num_worker,
        )

        # Add network latency to TTFT (KV transfer happens after prefill, before decode)
        if network_latency_ms > 0:
            disagg_summary_df['ttft'] = disagg_summary_df['ttft'] + network_latency_ms
            disagg_summary_df['request_latency'] = disagg_summary_df['request_latency'] + network_latency_ms
            logger.info(f"Added network latency {network_latency_ms:.3f}ms to TTFT")

        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary.set_summary_df(disagg_summary_df)
        
        # TODO: Store network info in summary for debugging
        # disagg_summary.network_info = {
        #     'kv_cache_size_bytes': kv_cache_size,
        #     'gpu_layout': gpu_layout,
        #     'network_latency_ms': network_latency_ms,
        # }
        
        return disagg_summary

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
                        )
                    else:
                        sess = InferenceSession(
                            model=model,
                            database=self._decode_database,
                            backend=self._decode_backend,
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
                raise RuntimeError(
                    f"No results found for any parallel configuration. Showing last exception: {exceptions[-1]}"
                ) from exceptions[-1]
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
        # set final disagg summary
        disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary
