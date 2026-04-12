#!/usr/bin/env python3
"""
Test script for run_disagg method with network simulation.
Place this file in: /scratch1/hanjiang/aiconfigurator/src/aiconfigurator/sdk/test_run_disagg.py
"""

import os
import sys

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.inference_session import DisaggInferenceSession
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend


def test_run_disagg():
    """Test run_disagg with network simulation."""
    print("=" * 60)
    print("Testing run_disagg with Network Simulation")
    print("=" * 60)

    # Model to test
    model_path = "Qwen/Qwen3-32B"
    
    # Runtime config
    runtime_config = config.RuntimeConfig(
        batch_size=1,  # Will be overridden by prefill/decode batch sizes
        beam_width=1,
        isl=4000,
        osl=500,
        prefix=0,
    )
    
    # System config
    system = "h200_sxm"
    backend_name = "trtllm"  # String for PerfDatabase (not TRTLLMBackend object)
    version = "1.2.0rc5"  # Add version
    
    # Create backend object for DisaggInferenceSession
    trtllm_backend = TRTLLMBackend()
    systems_dir = get_system_config_path()
    # Create database with string backend name
    database = PerfDatabase(system, backend_name, version, systems_dir=str(systems_dir))
    
    # Prefill model config (TP=1)
    prefill_model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        comm_quant_mode=common.CommQuantMode.half,
    )
    
    # Decode model config (TP=2)
    decode_model_config = config.ModelConfig(
        tp_size=2,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        comm_quant_mode=common.CommQuantMode.half,
    )
    
    # Network config path
    network_file = os.path.join(
        _PROJECT_ROOT, 
        "network_backend", 
        "astra-network-analytical", 
        "input", 
        "Ring.yml"
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  ISL: {runtime_config.isl}")
    print(f"  OSL: {runtime_config.osl}")
    print(f"  Prefill TP: {prefill_model_config.tp_size}")
    print(f"  Decode TP: {decode_model_config.tp_size}")
    print(f"  Network config: {network_file}")
    print(f"  Network config exists: {os.path.exists(network_file)}")
    
    # Test WITHOUT network simulation
    print("\n" + "-" * 60)
    print("Test 1: run_disagg WITHOUT network simulation")
    print("-" * 60)
    
    session_no_network = DisaggInferenceSession(
        prefill_database=database,
        prefill_backend=trtllm_backend,
        decode_database=database,
        decode_backend=trtllm_backend,
        network_file=None,  # Disable network simulation
    )
    
    summary_no_network = session_no_network.run_disagg(
        model_path=model_path,
        runtime_config=runtime_config,
        prefill_model_config=prefill_model_config,
        prefill_batch_size=1,
        prefill_num_worker=4,
        decode_model_config=decode_model_config,
        decode_batch_size=56,
        decode_num_worker=2,
    )
    
    df_no_network = summary_no_network.get_summary_df()
    print(f"\nResults (no network):")
    print(f"  TTFT: {df_no_network['ttft'].iloc[0]:.3f} ms")
    print(f"  TPOT: {df_no_network['tpot'].iloc[0]:.3f} ms")
    print(f"  Request latency: {df_no_network['request_latency'].iloc[0]:.3f} ms")
    print(f"  tokens/s/gpu: {df_no_network['tokens/s/gpu'].iloc[0]:.2f}")
    
    # Test WITH network simulation
    print("\n" + "-" * 60)
    print("Test 2: run_disagg WITH network simulation")
    print("-" * 60)
    
    if os.path.exists(network_file):
        session_with_network = DisaggInferenceSession(
            prefill_database=database,
            prefill_backend=trtllm_backend,
            decode_database=database,
            decode_backend=trtllm_backend,
            network_file=network_file,  # Enable network simulation
        )
        
        print(f"  AstraSim enabled: {session_with_network._enable_astrasim}")
        
        summary_with_network = session_with_network.run_disagg(
            model_path=model_path,
            runtime_config=runtime_config,
            prefill_model_config=prefill_model_config,
            prefill_batch_size=1,
            prefill_num_worker=4,
            decode_model_config=decode_model_config,
            decode_batch_size=56,
            decode_num_worker=2,
        )
        
        df_with_network = summary_with_network.get_summary_df()
        print(f"\nResults (with network):")
        print(f"  TTFT: {df_with_network['ttft'].iloc[0]:.3f} ms")
        print(f"  TPOT: {df_with_network['tpot'].iloc[0]:.3f} ms")
        print(f"  Request latency: {df_with_network['request_latency'].iloc[0]:.3f} ms")
        print(f"  tokens/s/gpu: {df_with_network['tokens/s/gpu'].iloc[0]:.2f}")
        
        # Check if network info was stored
        if hasattr(summary_with_network, 'network_info'):
            print(f"\nNetwork simulation info:")
            print(f"  KV cache size: {summary_with_network.network_info.get('kv_cache_size_bytes', 0) / 1e6:.2f} MB")
            print(f"  Network latency: {summary_with_network.network_info.get('kv_network_latency_ms', 0):.3f} ms")
        
        # Compare results
        print("\n" + "-" * 60)
        print("Comparison")
        print("-" * 60)
        ttft_no_net = df_no_network['ttft'].iloc[0]
        ttft_with_net = df_with_network['ttft'].iloc[0]
        network_overhead = ttft_with_net - ttft_no_net
        print(f"  TTFT without network: {ttft_no_net:.3f} ms")
        print(f"  TTFT with network:    {ttft_with_net:.3f} ms")
        print(f"  Network overhead:     {network_overhead:.3f} ms")
    else:
        print(f"  WARNING: Network config file not found: {network_file}")
        print("  Skipping network simulation test")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_run_disagg_simple():
    """Simplified test without network simulation."""
    print("=" * 60)
    print("Testing run_disagg (simple, no network)")
    print("=" * 60)
    
    model_path = "meta-llama/Llama-3.1-8B"
    
    runtime_config = config.RuntimeConfig(
        batch_size=1,
        beam_width=1,
        isl=2048,
        osl=256,
        prefix=0,
    )
    
    system = "h100_sxm"
    version = "v1"  # Add version
    backend = TRTLLMBackend()
    database = PerfDatabase(system, backend, version)
    
    model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=config.QuantMode.FP8_BLOCK,
        kvcache_quant_mode=config.QuantMode.FP8,
        fmha_mode=config.FMHAMode.FP8,
        moe_quant_mode=config.QuantMode.FP8_BLOCK,
        comm_quant_mode=config.QuantMode.HALF,
    )
    
    session = DisaggInferenceSession(
        prefill_database=database,
        prefill_backend=backend,
        decode_database=database,
        decode_backend=backend,
    )
    
    print(f"\nRunning run_disagg for {model_path}...")
    
    summary = session.run_disagg(
        model_path=model_path,
        runtime_config=runtime_config,
        prefill_model_config=model_config,
        prefill_batch_size=1,
        prefill_num_worker=2,
        decode_model_config=model_config,
        decode_batch_size=32,
        decode_num_worker=2,
    )
    
    df = summary.get_summary_df()
    print(f"\nResults:")
    print(df.to_string())
    
    print("\n" + "=" * 60)
    print("Simple test completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test run_disagg method")
    parser.add_argument("--simple", action="store_true", help="Run simple test only")
    args = parser.parse_args()
    
    if args.simple:
        test_run_disagg_simple()
    else:
        test_run_disagg()