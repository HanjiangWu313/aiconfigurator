# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MOEModel AFD (Attention-FFN Disaggregation) op classification.

Verifies that context_attn_ops, context_ffn_ops, generation_attn_ops, and
generation_ffn_ops are correctly populated after model construction.
"""

import pytest

from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import MOEModel
from aiconfigurator.sdk.operations import MoEDispatch
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit

# ---- helpers ----


class _CaptureAfdP2PDatabase:
    """Minimal database stub that records AFD transfer sizing."""

    def __init__(self):
        self.system_spec = {
            "gpu": {"sm_version": 100},
            "node": {"num_gpus_per_node": 8},
        }
        self.calls = []

    def query_afd_p2p(self, **kwargs):
        self.calls.append(kwargs)
        return PerformanceResult(0.0, energy=0.0)


def _expected_stage(op_name: str, model) -> str:
    """Return 'attn' or 'ffn' using the model's own classification logic."""
    return "attn" if model._is_attn_op(op_name) else "ffn"


def _build_moe_model(
    tp_size: int = 1,
    pp_size: int = 1,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
    num_experts: int = 8,
    topk: int = 2,
) -> MOEModel:
    """Construct a minimal MOEModel (Mixtral-like) for testing."""
    model_config = ModelConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=(tp_size * 1) // moe_tp_size * moe_tp_size // moe_ep_size if moe_ep_size > 1 else 1,
    )
    # Ensure the parallelism constraint: tp * adp == moe_tp * moe_ep
    model_config.attention_dp_size = (moe_tp_size * moe_ep_size) // tp_size

    # Mixtral-8x7B-like parameters
    moe_inter_size = 14336
    return MOEModel(
        topk,  # topk
        num_experts,  # num_experts
        moe_inter_size,  # moe_inter_size
        # BaseModel positional args:
        "mistralai/Mixtral-8x7B-v0.1",  # model_path
        "MOE",  # model_family
        "MixtralForCausalLM",  # architecture
        32,  # num_layers
        32,  # num_heads
        8,  # num_kv_heads
        128,  # head_size
        4096,  # hidden_size
        14336,  # inter_size
        32000,  # vocab_size
        32768,  # context_length
        model_config,
    )


# ---- tests ----


class TestMOEModelAFDOps:
    """Test that MOEModel correctly splits ops into attn/ffn lists for AFD."""

    def test_afd_lists_exist(self):
        """The four AFD lists should exist on every MOEModel instance."""
        model = _build_moe_model()
        assert hasattr(model, "context_attn_ops")
        assert hasattr(model, "context_ffn_ops")
        assert hasattr(model, "generation_attn_ops")
        assert hasattr(model, "generation_ffn_ops")

    def test_afd_lists_non_empty(self):
        """Each AFD list should contain at least one op."""
        model = _build_moe_model()
        assert len(model.context_attn_ops) > 0, "context_attn_ops should not be empty"
        assert len(model.context_ffn_ops) > 0, "context_ffn_ops should not be empty"
        assert len(model.generation_attn_ops) > 0, "generation_attn_ops should not be empty"
        assert len(model.generation_ffn_ops) > 0, "generation_ffn_ops should not be empty"

    def test_afd_union_equals_full_ops(self):
        """attn_ops + ffn_ops should cover all ops in context_ops / generation_ops."""
        model = _build_moe_model()

        ctx_attn_names = {op._name for op in model.context_attn_ops}
        ctx_ffn_names = {op._name for op in model.context_ffn_ops}
        ctx_all_names = {op._name for op in model.context_ops}

        # Union should equal the full set (note: duplicate names like "context_attention"
        # may appear when GptOssForCausalLM prepends an extra attention op)
        assert ctx_attn_names | ctx_ffn_names == ctx_all_names, (
            f"context attn/ffn union mismatch.\n"
            f"  missing from union: {ctx_all_names - (ctx_attn_names | ctx_ffn_names)}\n"
            f"  extra in union:     {(ctx_attn_names | ctx_ffn_names) - ctx_all_names}"
        )

        gen_attn_names = {op._name for op in model.generation_attn_ops}
        gen_ffn_names = {op._name for op in model.generation_ffn_ops}
        gen_all_names = {op._name for op in model.generation_ops}

        assert gen_attn_names | gen_ffn_names == gen_all_names, (
            f"generation attn/ffn union mismatch.\n"
            f"  missing from union: {gen_all_names - (gen_attn_names | gen_ffn_names)}\n"
            f"  extra in union:     {(gen_attn_names | gen_ffn_names) - gen_all_names}"
        )

    def test_afd_no_overlap(self):
        """attn_ops and ffn_ops should have no overlap (by object identity)."""
        model = _build_moe_model()

        ctx_attn_ids = {id(op) for op in model.context_attn_ops}
        ctx_ffn_ids = {id(op) for op in model.context_ffn_ops}
        assert ctx_attn_ids.isdisjoint(ctx_ffn_ids), "context attn and ffn ops should not overlap"

        gen_attn_ids = {id(op) for op in model.generation_attn_ops}
        gen_ffn_ids = {id(op) for op in model.generation_ffn_ops}
        assert gen_attn_ids.isdisjoint(gen_ffn_ids), "generation attn and ffn ops should not overlap"

    def test_afd_count_matches_full_ops(self):
        """Total number of ops in attn + ffn should match context_ops / generation_ops."""
        model = _build_moe_model()

        assert len(model.context_attn_ops) + len(model.context_ffn_ops) == len(model.context_ops), (
            f"context count mismatch: {len(model.context_attn_ops)} attn + "
            f"{len(model.context_ffn_ops)} ffn != {len(model.context_ops)} total"
        )
        assert len(model.generation_attn_ops) + len(model.generation_ffn_ops) == len(model.generation_ops), (
            f"generation count mismatch: {len(model.generation_attn_ops)} attn + "
            f"{len(model.generation_ffn_ops)} ffn != {len(model.generation_ops)} total"
        )

    def test_attn_ops_contain_expected_names(self):
        """Attention ops should include embedding, qkv_gemm, attention, proj_gemm, etc."""
        model = _build_moe_model()

        ctx_attn_names = [op._name for op in model.context_attn_ops]
        assert "context_embedding" in ctx_attn_names
        assert "context_qkv_gemm" in ctx_attn_names
        assert "context_attention" in ctx_attn_names
        assert "context_proj_gemm" in ctx_attn_names
        assert "context_add_norm_1" in ctx_attn_names
        assert "context_p2p" in ctx_attn_names

        gen_attn_names = [op._name for op in model.generation_attn_ops]
        assert "generation_embedding" in gen_attn_names
        assert "generation_qkv_gemm" in gen_attn_names
        assert "generation_attention" in gen_attn_names
        assert "generation_proj_gemm" in gen_attn_names
        assert "generation_add_norm_1" in gen_attn_names
        assert "generation_p2p" in gen_attn_names

    def test_ffn_ops_contain_expected_names(self):
        """FFN ops should include add_norm_2, moe dispatch, moe compute, logits, etc."""
        model = _build_moe_model()

        ctx_ffn_names = [op._name for op in model.context_ffn_ops]
        assert "context_add_norm_2" in ctx_ffn_names
        assert "context_moe_pre_dispatch" in ctx_ffn_names
        assert "context_moe" in ctx_ffn_names
        assert "context_moe_post_dispatch" in ctx_ffn_names

        gen_ffn_names = [op._name for op in model.generation_ffn_ops]
        assert "generation_add_norm_2" in gen_ffn_names
        assert "generation_moe_pre_dispatch" in gen_ffn_names
        assert "generation_moe" in gen_ffn_names
        assert "generation_moe_post_dispatch" in gen_ffn_names
        assert "generation_logits_gemm" in gen_ffn_names

    def test_attn_ops_exclude_ffn_names(self):
        """Attention op lists should not contain any FFN-stage op names."""
        model = _build_moe_model()

        ffn_keywords = {"moe", "router", "logits", "add_norm_2"}
        for op in model.context_attn_ops:
            stripped = op._name.replace("context_", "")
            assert stripped not in ffn_keywords and not stripped.startswith("moe"), (
                f"FFN-stage op '{op._name}' found in context_attn_ops"
            )

    def test_classification_matches_model_is_attn_op(self):
        """Every op should be classified identically to model._is_attn_op()."""
        model = _build_moe_model()

        for op in model.context_ops:
            expected = _expected_stage(op._name, model)
            if expected == "attn":
                assert op in model.context_attn_ops, f"'{op._name}' should be in context_attn_ops (expected stage=attn)"
            else:
                assert op in model.context_ffn_ops, f"'{op._name}' should be in context_ffn_ops (expected stage=ffn)"

        for op in model.generation_ops:
            expected = _expected_stage(op._name, model)
            if expected == "attn":
                assert op in model.generation_attn_ops, (
                    f"'{op._name}' should be in generation_attn_ops (expected stage=attn)"
                )
            else:
                assert op in model.generation_ffn_ops, (
                    f"'{op._name}' should be in generation_ffn_ops (expected stage=ffn)"
                )

    def test_large_expert_count_includes_router(self):
        """When num_experts >= 128, router_gemm should appear in attn_ops."""
        model = _build_moe_model(num_experts=256)

        ctx_attn_names = [op._name for op in model.context_attn_ops]
        gen_attn_names = [op._name for op in model.generation_attn_ops]
        assert "context_router_gemm" in ctx_attn_names, "router_gemm missing from context_attn_ops"
        assert "generation_router_gemm" in gen_attn_names, "router_gemm missing from generation_attn_ops"

    def test_small_expert_count_no_router(self):
        """When num_experts < 128, router_gemm should NOT appear."""
        model = _build_moe_model(num_experts=8)

        all_names = [op._name for op in model.context_ops] + [op._name for op in model.generation_ops]
        assert "context_router_gemm" not in all_names
        assert "generation_router_gemm" not in all_names

    def test_pp_size_does_not_break_classification(self):
        """Pipeline parallelism should not affect AFD classification."""
        model = _build_moe_model(pp_size=2)

        # p2p should still be in attn
        ctx_attn_names = [op._name for op in model.context_attn_ops]
        assert "context_p2p" in ctx_attn_names

        # Counts should still match
        assert len(model.context_attn_ops) + len(model.context_ffn_ops) == len(model.context_ops)


class TestMoEDispatchAFDTransferSizing:
    """Validate AFD transfer bytes use attention DP, not total attention GPUs."""

    def test_pre_dispatch_sender_bytes_are_tp_aware(self):
        op = MoEDispatch(
            "context_moe_pre_dispatch",
            scale_factor=1,
            hidden_size=2048,
            topk=8,
            num_experts=128,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=2,
            pre_dispatch=True,
            enable_afd=True,
        )
        db = _CaptureAfdP2PDatabase()

        op.query(db, x=256, num_attn_gpus=4, num_ffn_gpus=4)

        call = db.calls[-1]
        assert call["sender_bytes"] == 1_310_720
        assert call["receiver_bytes"] == 1_310_720
        assert call["num_gpus"] == 8

    def test_post_dispatch_sender_bytes_are_tp_aware(self):
        op = MoEDispatch(
            "context_moe_post_dispatch",
            scale_factor=1,
            hidden_size=2048,
            topk=8,
            num_experts=128,
            moe_tp_size=1,
            moe_ep_size=4,
            attention_dp_size=2,
            pre_dispatch=False,
            enable_afd=True,
        )
        db = _CaptureAfdP2PDatabase()

        op.query(db, x=256, num_attn_gpus=4, num_ffn_gpus=4)

        call = db.calls[-1]
        assert call["sender_bytes"] == 524_288
        assert call["receiver_bytes"] == 524_288
        assert call["num_gpus"] == 8
