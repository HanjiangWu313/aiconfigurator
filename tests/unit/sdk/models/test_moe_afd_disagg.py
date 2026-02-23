# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for MOEModel AFD in DisaggInferenceSession.

Verifies that ``run_disagg_afd`` correctly produces attention / FFN latency
breakdowns for both prefill (context) and decode (generation) workers in a
disaggregated serving setup.

"""

import pytest

from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import DisaggInferenceSession
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.common import DatabaseMode
pytestmark = pytest.mark.unit

# Real HuggingFace model path — config is pre-downloaded in model_configs/
_MODEL_PATH = "Qwen/Qwen3-30B-A3B"
_SYSTEM = "b200_sxm"
_BACKEND = "trtllm"
_VERSION = "1.2.0rc5"


def _classify_op_stage(op_name: str) -> str:
    """Classify an op as 'attn' or 'ffn' using BaseModel's default classification."""
    # Use BaseModel's default _ATTN_COMPONENTS (MOEModel inherits the same set)
    name = op_name
    for pfx in ("context_", "generation_"):
        if name.startswith(pfx):
            name = name[len(pfx):]
            break
    return "attn" if name in BaseModel._ATTN_COMPONENTS else "ffn"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perf_db():
    """Real PerfDatabase reading from actual system perf data."""
    systems_dir = get_system_config_path()
    return PerfDatabase(_SYSTEM, _BACKEND, _VERSION, systems_dir=str(systems_dir))


@pytest.fixture(autouse=True)
def _clear_model_info_cache():
    """Clear the _get_model_info LRU cache between tests."""
    import aiconfigurator.sdk.models as models_module

    models_module._get_model_info.cache_clear()
    yield
    models_module._get_model_info.cache_clear()


@pytest.fixture
def prefill_model_config():
    return ModelConfig(
        tp_size=1, pp_size=1,
        moe_tp_size=1, moe_ep_size=1, attention_dp_size=1,
    )


@pytest.fixture
def decode_model_config():
    return ModelConfig(
        tp_size=1, pp_size=1,
        moe_tp_size=1, moe_ep_size=1, attention_dp_size=1,
    )


@pytest.fixture
def runtime_cfg():
    return RuntimeConfig(batch_size=128, isl=6400, osl=1000, beam_width=1)


@pytest.fixture
def disagg_session(perf_db):
    """Create a DisaggInferenceSession using the same DB for prefill and decode."""
    
    # Create a single TensorRT LLM backend
    backend = TRTLLMBackend()
    
    return DisaggInferenceSession(
        prefill_database=perf_db,
        prefill_backend=backend,
        decode_database=perf_db,
        decode_backend=backend,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDisaggAFDBreakdown:
    """
    Verify that ``run_disagg_afd`` produces correct attention / FFN latency
    breakdowns for both prefill and decode workers.
    """

    def _run_afd(self, session, runtime_cfg, prefill_mc, decode_mc):
        return session.run_disagg_afd(
            model_path=_MODEL_PATH,
            runtime_config=runtime_cfg,
            prefill_model_config=prefill_mc,
            prefill_batch_size=2,
            prefill_num_worker=1,
            decode_model_config=decode_mc,
            decode_batch_size=2,
            decode_num_worker=1,
        )


    def test_prefill_attn_positive_latency(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Prefill attention phase should have positive latency."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        ctx_dict = result["prefill_attn"].get_context_latency_dict()
        assert len(ctx_dict) > 0
        assert sum(ctx_dict.values()) > 0

    def test_prefill_ffn_positive_latency(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Prefill FFN phase should have positive latency."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        ctx_dict = result["prefill_ffn"].get_context_latency_dict()
        assert len(ctx_dict) > 0
        assert sum(ctx_dict.values()) > 0

    def test_decode_attn_positive_latency(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Decode attention phase should have positive latency."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        gen_dict = result["decode_attn"].get_generation_latency_dict()
        assert len(gen_dict) > 0
        assert sum(gen_dict.values()) > 0

    def test_decode_ffn_positive_latency(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Decode FFN phase should have positive latency."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        gen_dict = result["decode_ffn"].get_generation_latency_dict()
        assert len(gen_dict) > 0
        assert sum(gen_dict.values()) > 0

    # ---- additivity: attn + ffn == full ----

    def test_prefill_attn_plus_ffn_equals_full(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Prefill attn latency + prefill ffn latency should equal full prefill latency."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        full_total = sum(result["prefill_full"].get_context_latency_dict().values())
        attn_total = sum(result["prefill_attn"].get_context_latency_dict().values())
        ffn_total = sum(result["prefill_ffn"].get_context_latency_dict().values())

        assert full_total > 0
        assert abs(attn_total + ffn_total - full_total) / full_total < 1e-6, (
            f"prefill attn ({attn_total:.4f}) + ffn ({ffn_total:.4f}) != "
            f"full ({full_total:.4f})"
        )

    def test_decode_attn_plus_ffn_equals_full(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Decode attn latency + decode ffn latency should equal full decode latency."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        full_total = sum(result["decode_full"].get_generation_latency_dict().values())
        attn_total = sum(result["decode_attn"].get_generation_latency_dict().values())
        ffn_total = sum(result["decode_ffn"].get_generation_latency_dict().values())

        assert full_total > 0
        assert abs(attn_total + ffn_total - full_total) / full_total < 1e-6, (
            f"decode attn ({attn_total:.4f}) + ffn ({ffn_total:.4f}) != "
            f"full ({full_total:.4f})"
        )

    
    
    def test_returns_all_time_breakdown(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):  
        """run_disagg_afd should return a dict with all documented keys.

        Also prints a side-by-side comparison of SILICON (measured) vs SOL
        (speed-of-light) latencies so we can see how close the real data is
        to the theoretical roofline.
        """
        
        # The default results is generated in silicon
        result_silicon = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )

        # --- Switch both databases to SOL and run again ---
        prefill_db = disagg_session._prefill_database
        decode_db = disagg_session._decode_database
        prefill_db.set_default_database_mode(DatabaseMode.SOL)
        decode_db.set_default_database_mode(DatabaseMode.SOL)
        result_sol = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )

        # --- Switch both databases to EMPIRICAL and run again ---
        prefill_db.set_default_database_mode(DatabaseMode.EMPIRICAL)
        decode_db.set_default_database_mode(DatabaseMode.EMPIRICAL)
        result_emp = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )

        # Restore to SILICON so other tests are not affected
        prefill_db.set_default_database_mode(DatabaseMode.SILICON)
        decode_db.set_default_database_mode(DatabaseMode.SILICON)

        # --- Print side-by-side comparison ---
        for label, key, method in [
            ("PREFILL ATTN", "prefill_attn", "get_context_latency_dict"),
            ("PREFILL FFN",  "prefill_ffn",  "get_context_latency_dict"),
            ("DECODE ATTN",  "decode_attn",  "get_generation_latency_dict"),
            ("DECODE FFN",   "decode_ffn",   "get_generation_latency_dict"),
        ]:
            ops_si = getattr(result_silicon[key], method)()
            ops_sol = getattr(result_sol[key], method)()
            ops_emp = getattr(result_emp[key], method)()
            total_si = sum(ops_si.values())
            total_sol = sum(ops_sol.values())
            total_emp = sum(ops_emp.values())

            print(f"\n{'=' * 110}")
            print(f"  {label}  (SILICON={total_si:.4f} ms,  SOL={total_sol:.4f} ms,  EMPIRICAL={total_emp:.4f} ms)")
            print(f"  {'op':<34s}  {'SILICON':>10s}  {'SOL':>10s}  {'EMPIRICAL':>10s}  {'SI/SOL':>8s}  {'SI/EMP':>8s}")
            print(f"  {'-' * 104}")

            all_ops = list(ops_si.keys())
            
            # Compare each operators between SI, SOL, and Empirical
            for op in all_ops:
                if op == "generation_moe_pre_dispatch":
                    print(op, float(ops_si.get(op)), float(ops_sol.get(op)), float(ops_emp.get(op)))
                si = ops_si.get(op, 0.0000)
                sol = ops_sol.get(op, 0.0000)
                emp = ops_emp.get(op, 0.0000)
                ratio_sol = si / sol if sol > 0 else float("inf")
                ratio_emp = si / emp if emp > 0 else float("inf")
                print(f"  {op:<34s}  {si:10.4f}  {sol:10.4f}  {emp:10.4f}  {ratio_sol:8.2f}x  {ratio_emp:8.2f}x")

            # Get the total time breakdown
            overall_sol = total_si / total_sol if total_sol > 0 else float("inf")
            overall_emp = total_si / total_emp if total_emp > 0 else float("inf")
            print(f"  {'TOTAL':<34s}  {total_si:10.4f}  {total_sol:10.4f}  {total_emp:10.4f}  {overall_sol:8.2f}x  {overall_emp:8.2f}x")

        expected_keys = {
            "disagg_summary",
            "prefill_full", "prefill_attn", "prefill_ffn",
            "decode_full", "decode_attn", "decode_ffn",
            "prefill_attn_pct", "prefill_ffn_pct",
            "decode_attn_pct", "decode_ffn_pct",
        }
        assert expected_keys == set(result_silicon.keys())
    
    # ---- op-name classification ----

    def test_prefill_attn_ops_are_attn_stage(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """All ops in prefill_attn result should be classified as 'attn'."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        for op_name in result["prefill_attn"].get_context_latency_dict():
            stage = _classify_op_stage(op_name)
            assert stage == "attn", (
                f"Op '{op_name}' in prefill_attn classified as '{stage}'"
            )

    def test_prefill_ffn_ops_are_ffn_stage(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """All ops in prefill_ffn result should be classified as 'ffn'."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        for op_name in result["prefill_ffn"].get_context_latency_dict():
            stage = _classify_op_stage(op_name)
            assert stage == "ffn", (
                f"Op '{op_name}' in prefill_ffn classified as '{stage}'"
            )

    def test_decode_attn_ops_are_attn_stage(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """All ops in decode_attn result should be classified as 'attn'."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        for op_name in result["decode_attn"].get_generation_latency_dict():
            stage = _classify_op_stage(op_name)
            assert stage == "attn", (
                f"Op '{op_name}' in decode_attn classified as '{stage}'"
            )

    def test_decode_ffn_ops_are_ffn_stage(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """All ops in decode_ffn result should be classified as 'ffn'."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        for op_name in result["decode_ffn"].get_generation_latency_dict():
            stage = _classify_op_stage(op_name)
            assert stage == "ffn", (
                f"Op '{op_name}' in decode_ffn classified as '{stage}'"
            )

    # ---- percentage breakdowns ----

    def test_percentages_are_meaningful(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Each AFD percentage should be > 0% and < 100%, and attn + ffn ≈ 100%."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        p_attn = result["prefill_attn_pct"]
        p_ffn = result["prefill_ffn_pct"]
        d_attn = result["decode_attn_pct"]
        d_ffn = result["decode_ffn_pct"]

        assert 0 < p_attn < 100, f"prefill attn% = {p_attn:.1f}% out of range"
        assert 0 < p_ffn < 100, f"prefill ffn%  = {p_ffn:.1f}% out of range"
        assert 0 < d_attn < 100, f"decode attn%  = {d_attn:.1f}% out of range"
        assert 0 < d_ffn < 100, f"decode ffn%   = {d_ffn:.1f}% out of range"

        assert abs(p_attn + p_ffn - 100) < 0.01, (
            f"prefill attn% + ffn% = {p_attn + p_ffn:.2f}% != 100%"
        )
        assert abs(d_attn + d_ffn - 100) < 0.01, (
            f"decode attn% + ffn% = {d_attn + d_ffn:.2f}% != 100%"
        )

    # ---- no cross-contamination ----

    def test_prefill_attn_has_no_ffn_ops(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """prefill_attn should NOT contain moe/router/logits ops."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        ffn_keywords = {"moe", "router", "logits", "add_norm_2"}
        for op_name in result["prefill_attn"].get_context_latency_dict():
            stripped = op_name.replace("context_", "")
            assert stripped not in ffn_keywords and not stripped.startswith("moe"), (
                f"FFN op '{op_name}' leaked into prefill_attn"
            )

    def test_decode_ffn_has_no_attn_ops(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """decode_ffn should NOT contain embedding/qkv/attention/proj ops."""
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        attn_keywords = {"embedding", "qkv_gemm", "attention", "proj_gemm", "add_norm_1"}
        for op_name in result["decode_ffn"].get_generation_latency_dict():
            stripped = op_name.replace("generation_", "")
            assert stripped not in attn_keywords, (
                f"Attn op '{op_name}' leaked into decode_ffn"
            )

    # ---- consistency between run_disagg and run_disagg_afd ----

    def test_full_latencies_match_run_disagg(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """
        The full prefill/decode latencies from run_disagg_afd should match
        what a standalone run_disagg would produce (since they use the same
        underlying run_static calls).
        """
        afd_result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )

        # run_disagg separately
        standalone = disagg_session.run_disagg(
            model_path=_MODEL_PATH,
            runtime_config=runtime_cfg,
            prefill_model_config=prefill_model_config,
            prefill_batch_size=2,
            prefill_num_worker=1,
            decode_model_config=decode_model_config,
            decode_batch_size=2,
            decode_num_worker=1,
        )

        # Compare TTFT and TPOT from summary dataframes
        afd_df = afd_result["disagg_summary"].get_summary_df()
        std_df = standalone.get_summary_df()

        assert abs(afd_df["ttft"].iloc[0] - std_df["ttft"].iloc[0]) < 1e-3, (
            f"TTFT mismatch: afd={afd_df['ttft'].iloc[0]:.4f} vs "
            f"standalone={std_df['ttft'].iloc[0]:.4f}"
        )
        assert abs(afd_df["tpot"].iloc[0] - std_df["tpot"].iloc[0]) < 1e-3, (
            f"TPOT mismatch: afd={afd_df['tpot'].iloc[0]:.4f} vs "
            f"standalone={std_df['tpot'].iloc[0]:.4f}"
        )
