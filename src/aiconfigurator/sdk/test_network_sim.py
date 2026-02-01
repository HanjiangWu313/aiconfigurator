#!/usr/bin/env python3
"""
Test script for AstraSim network simulator integration.
"""

import os
import sys

# Get the directory of this file to build relative paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

# Default network config path (relative to project root)
DEFAULT_NETWORK_CONFIG = os.path.join(_PROJECT_ROOT, "network_backend", "astra-network-analytical", "input", "Ring.yml")

# Network simulator library path (relative to project root)
_NETWORK_SIM_LIB_PATH = os.path.join(_PROJECT_ROOT, "network_backend", "astra-network-analytical", "lib")

print(f"Project root: {_PROJECT_ROOT}")
print(f"Network config: {DEFAULT_NETWORK_CONFIG}")
print(f"Network lib path: {_NETWORK_SIM_LIB_PATH}")
print(f"Config exists: {os.path.exists(DEFAULT_NETWORK_CONFIG)}")
print(f"Lib path exists: {os.path.exists(_NETWORK_SIM_LIB_PATH)}")

if os.path.exists(_NETWORK_SIM_LIB_PATH):
    print(f"Lib contents: {os.listdir(_NETWORK_SIM_LIB_PATH)}")

# Add network simulator to path
sys.path.append(_NETWORK_SIM_LIB_PATH)
try:
    from simulation_py_congestion_aware import (
        EventQueue,
        Topology,
        Chunk,
        NetworkParser,
        construct_topology,
    )
    print("\n✓ Successfully imported AstraSim network simulator!")
    NETWORK_SIM_AVAILABLE = True
except ImportError as e:
    print(f"\n✗ Failed to import network simulator: {e}")
    import traceback
    traceback.print_exc()
    NETWORK_SIM_AVAILABLE = False
    sys.exit(1)


def run_simulation():
    """Run a simple network simulation test."""
    print("\n" + "=" * 60)
    print("Running AstraSim Network Simulation Test")
    print("=" * 60)
    
    chunk_size = 1048576  # 1MB
    
    # Create network parser
    print(f"\nLoading network config: {DEFAULT_NETWORK_CONFIG}")
    network_parser = NetworkParser(DEFAULT_NETWORK_CONFIG)
    
    # Create event queue
    event_queue = EventQueue()
    Topology.set_event_queue(event_queue)
    topology = construct_topology(network_parser)
    
    npus_count = topology.get_npus_count()
    devices_count = topology.get_devices_count()
    
    print(f"Number of NPUs in the topology: {npus_count}")
    print(f"Number of devices: {devices_count}")
    
    # Simulate AllGather-like transfers
    request_id = 0
    print(f"\nSending {npus_count * (devices_count - 1)} chunks of {chunk_size} bytes each...")
    
    for i in range(npus_count):
        for j in range(devices_count):
            if i == j:
                continue
            chunk = Chunk.create_with_event_queue(
                chunk_size, i, j, request_id, topology, event_queue
            )
            topology.send_python(chunk)
            request_id += 1
    
    print(f"Total chunks sent: {request_id}")
    
    # Process all events
    print("\nProcessing network events...")
    iteration = 0
    max_iterations = 100000  # Safety limit
    while not event_queue.finished() and iteration < max_iterations:
        event_queue.proceed()
        arrivals = event_queue.get_and_clear_arrivals()
        if arrivals:
            print(f"  Iteration {iteration}: Arrived req_ids: {arrivals}")
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"  WARNING: Reached max iterations ({max_iterations})")
    
    final_time = event_queue.get_current_time()
    print(f"\n✓ Simulation finished!")
    print(f"  End time: {final_time} ns")
    print(f"  End time: {final_time / 1e6:.3f} ms")
    print(f"  End time: {final_time / 1e9:.6f} s")
    
    return final_time


def test_kv_cache_transfer():
    """Test simulating KV cache transfer between prefill and decode GPUs."""
    print("\n" + "=" * 60)
    print("Testing KV Cache Transfer Simulation")
    print("=" * 60)
    
    # First, check topology size
    network_parser = NetworkParser(DEFAULT_NETWORK_CONFIG)
    event_queue = EventQueue()
    Topology.set_event_queue(event_queue)
    topology = construct_topology(network_parser)
    
    npus_count = topology.get_npus_count()
    devices_count = topology.get_devices_count()
    
    print(f"\nTopology info:")
    print(f"  NPUs count: {npus_count}")
    print(f"  Devices count: {devices_count}")
    
    # Adjust GPU IDs based on topology size
    # Use first half as prefill, second half as decode
    if devices_count < 4:
        print(f"  WARNING: Topology has only {devices_count} devices, using smaller test")
        prefill_gpus = [0]
        decode_gpus = [1] if devices_count > 1 else [0]
    else:
        half = devices_count // 2
        prefill_gpus = list(range(half))[:4]  # Max 4 prefill GPUs
        decode_gpus = list(range(half, devices_count))[:2]  # Max 2 decode GPUs
    
    print(f"  Using prefill GPUs: {prefill_gpus}")
    print(f"  Using decode GPUs: {decode_gpus}")
    
    # KV cache size: batch_size * isl * hidden_size * dtype_size
    # Use smaller size for testing
    batch_size = 1
    isl = 4000
    hidden_size = 5120
    dtype_size = 2  # fp16
    kv_cache_size = batch_size * isl * hidden_size * dtype_size
    
    print(f"\nKV Cache Transfer Parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  ISL: {isl}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Dtype size: {dtype_size} bytes")
    print(f"  Total KV cache size: {kv_cache_size / 1e6:.2f} MB")
    
    # Reset network for fresh simulation
    event_queue = EventQueue()
    Topology.set_event_queue(event_queue)
    topology = construct_topology(network_parser)
    
    # Send KV cache from each prefill GPU to a decode GPU (round-robin)
    print("\nSending KV cache transfers:")
    chunks_sent = 0
    for i, src_gpu in enumerate(prefill_gpus):
        dst_gpu = decode_gpus[i % len(decode_gpus)]
        print(f"  Chunk {i}: GPU {src_gpu} -> GPU {dst_gpu}: {kv_cache_size / 1e6:.2f} MB")
        
        chunk = Chunk.create_with_event_queue(
            kv_cache_size, src_gpu, dst_gpu, i, topology, event_queue
        )
        topology.send_python(chunk)
        chunks_sent += 1
    
    print(f"\nTotal chunks sent: {chunks_sent}")
    print("Processing network events...")
    
    # Process events with timeout
    iteration = 0
    max_iterations = 1000000  # Safety limit
    while not event_queue.finished() and iteration < max_iterations:
        event_queue.proceed()
        arrivals = event_queue.get_and_clear_arrivals()
        if arrivals and iteration % 10000 == 0:
            print(f"  Iteration {iteration}: Arrived req_ids: {arrivals}")
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"  WARNING: Reached max iterations ({max_iterations}), simulation may be incomplete")
        final_time = event_queue.get_current_time()
    else:
        final_time = event_queue.get_current_time()
    
    print(f"\n✓ KV Cache Transfer Simulation Complete!")
    print(f"  Iterations: {iteration}")
    print(f"  Network latency: {final_time} ns")
    print(f"  Network latency: {final_time / 1e6:.3f} ms")
    print(f"  Network latency: {final_time / 1e9:.6f} s")
    
    # Calculate effective bandwidth
    total_bytes = kv_cache_size * chunks_sent
    if final_time > 0:
        bandwidth_gbps = (total_bytes * 8) / final_time  # bits per ns = Gbps
        print(f"  Total data transferred: {total_bytes / 1e6:.2f} MB")
        print(f"  Effective bandwidth: {bandwidth_gbps:.2f} Gbps")
    
    return final_time


def test_small_transfer():
    """Test with a small transfer to verify basic functionality."""
    print("\n" + "=" * 60)
    print("Testing Small Transfer (1 KB)")
    print("=" * 60)
    
    network_parser = NetworkParser(DEFAULT_NETWORK_CONFIG)
    event_queue = EventQueue()
    Topology.set_event_queue(event_queue)
    topology = construct_topology(network_parser)
    
    npus_count = topology.get_npus_count()
    devices_count = topology.get_devices_count()
    
    print(f"Topology: {npus_count} NPUs, {devices_count} devices")
    
    # Small 1 KB transfer
    chunk_size = 1024  # 1 KB
    src_gpu = 0
    dst_gpu = 1 if devices_count > 1 else 0
    
    print(f"Sending 1 KB from GPU {src_gpu} to GPU {dst_gpu}...")
    
    chunk = Chunk.create_with_event_queue(
        chunk_size, src_gpu, dst_gpu, 0, topology, event_queue
    )
    topology.send_python(chunk)
    
    # Process with progress
    iteration = 0
    max_iterations = 10000
    while not event_queue.finished() and iteration < max_iterations:
        event_queue.proceed()
        iteration += 1
        if iteration % 1000 == 0:
            print(f"  Iteration {iteration}, time: {event_queue.get_current_time()} ns")
    
    final_time = event_queue.get_current_time()
    print(f"\n✓ Small transfer complete!")
    print(f"  Iterations: {iteration}")
    print(f"  Latency: {final_time} ns = {final_time / 1e3:.2f} µs")
    
    return final_time


if __name__ == "__main__":
    print("AstraSim Network Simulator Integration Test")
    print("=" * 60)
    
    # Test small transfer first
    test_small_transfer()
    
    # Run basic simulation
    run_simulation()
    
    # Run KV cache transfer test
    test_kv_cache_transfer()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)