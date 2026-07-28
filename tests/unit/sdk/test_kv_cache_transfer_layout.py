# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk.astrasim_utils import AstraSimManager
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import DisaggInferenceSession

pytestmark = pytest.mark.unit


_SYSTEM_SPEC = {
    "node": {
        "num_gpus_per_node": 8,
        "intra_node_bw": 900e9,
        "inter_node_bw": 100e9,
        "p2p_latency": 1e-6,
    },
    "misc": {
        "nccl_mem": {1: 0, 2: 0, 4: 0, 8: 0},
        "other_mem": 0,
    },
}


class _DummyBackend:
    name = SimpleNamespace(value="dummy")

    def get_total_kv_cache_transfer_size_bytes(
        self,
        model,
        batch_size: int,
        isl: int,
        beam_width: int = 1,
        osl: int = 0,
    ) -> int:
        return batch_size * isl * 10


def _build_session() -> DisaggInferenceSession:
    backend = _DummyBackend()
    database = SimpleNamespace(system_spec=_SYSTEM_SPEC)
    return DisaggInferenceSession(
        prefill_database=database,
        prefill_backend=backend,
        decode_database=database,
        decode_backend=backend,
        astrasim_manager=AstraSimManager(system_spec=_SYSTEM_SPEC),
    )


def test_compute_kv_cache_transfer_size_scales_with_attention_dp():
    session = _build_session()
    runtime_config = RuntimeConfig(batch_size=2, isl=8, osl=1)
    model = SimpleNamespace(config=SimpleNamespace(attention_dp_size=4))

    transfer_size = session._compute_kv_cache_transfer_size(
        prefill_model=model,
        runtime_config=runtime_config,
        prefill_batch_size=runtime_config.batch_size,
    )

    assert transfer_size == 2 * 8 * 10 * 4


def test_build_gpu_layout_tracks_attention_dp_for_shared_moe_workers():
    session = _build_session()
    model_config = ModelConfig(
        tp_size=2,
        pp_size=2,
        moe_tp_size=1,
        moe_ep_size=4,
        attention_dp_size=2,
        enable_afd=False,
    )

    layout = session._build_gpu_layout(
        prefill_model_config=model_config,
        prefill_num_worker=1,
        decode_model_config=model_config,
        decode_num_worker=0,
    )

    worker = layout["prefill_worker_layouts"][0]

    assert layout["prefill_gpus_per_worker"] == 8
    assert worker["gpu_ids"] == list(range(8))
    assert worker["pp_stages"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert worker["attn_dp_pp_stages"] == [
        [[0, 1], [4, 5]],
        [[2, 3], [6, 7]],
    ]
    assert worker["attn_dp_size"] == 2
    assert worker["ffn_tp_size"] == 1
    assert worker["ffn_ep_size"] == 4


def test_build_kv_transfer_plan_uses_all_shared_worker_dp_replicas():
    session = _build_session()
    manager = session._astrasim
    model_config = ModelConfig(
        tp_size=1,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=2,
        attention_dp_size=2,
        enable_afd=False,
    )

    layout = session._build_gpu_layout(
        prefill_model_config=model_config,
        prefill_num_worker=1,
        decode_model_config=model_config,
        decode_num_worker=1,
    )
    transfers = manager.build_kv_transfer_plan(
        gpu_layout=layout,
        kv_cache_size=100,
        prefill_batch_size=4,
    )

    assert sum(t["bytes"] for t in transfers) == 100
    assert {(t["src"], t["dst"]) for t in transfers} == {(0, 2), (1, 3)}
    assert {t["src_dp_rank"] for t in transfers} == {0, 1}


def test_build_kv_transfer_plan_keeps_afd_transfers_on_attention_gpus():
    session = _build_session()
    manager = session._astrasim
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
        prefill_num_worker=1,
        decode_model_config=model_config,
        decode_num_worker=1,
    )
    transfers = manager.build_kv_transfer_plan(
        gpu_layout=layout,
        kv_cache_size=100,
        prefill_batch_size=4,
    )

    assert sum(t["bytes"] for t in transfers) == 100
    assert {(t["src"], t["dst"]) for t in transfers} == {(0, 4), (1, 5)}
    assert all(t["src"] in {0, 1} for t in transfers)
    assert all(t["dst"] in {4, 5} for t in transfers)


def test_prepare_inter_node_group_densely_remaps_global_nic_endpoints():
    manager = AstraSimManager(system_spec=_SYSTEM_SPEC)
    group = [
        (0, 8, 100),
        (4, 8, 100),
        (4, 9, 100),
        (8, 16, 100),
    ]

    remapped, num_units, participant_to_local = manager._prepare_group_for_topology(
        "inter-node",
        group,
    )

    assert remapped == [
        (0, 2, 100),
        (1, 2, 100),
        (1, 3, 100),
        (2, 4, 100),
    ]
    assert num_units == 5
    assert participant_to_local == {
        0: 0,
        4: 1,
        8: 2,
        9: 3,
        16: 4,
    }
