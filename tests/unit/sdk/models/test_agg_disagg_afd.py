# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modular tests for Agg, Disagg, AFD breakdowns

Running cmd for breakdown tests (from the root of the aiconfigurator repo):

Agg:
python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestAggBreakdown::test_returns_all_time_breakdown -s

Disagg:
python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestDisaggBreakdown::test_returns_all_time_breakdown -s

Disagg AFD:
python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestDisaggAFDBreakdown::test_returns_all_time_breakdown -s

"""

from pyexpat import model

import pytest

from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import DisaggInferenceSession, InferenceSession
from aiconfigurator.sdk.models import BaseModel, get_model
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.common import DatabaseMode
pytestmark = pytest.mark.unit

## Follows the naming split with "/" like huggingface, list of models is in here: 
## /scratch1/hanjiang/aiconfigurator/src/aiconfigurator/model_configs
_MODEL_PATH = "Qwen/Qwen3-30B-A3B"
## The system matches the yaml file in dir ,for example here is b200_sxm 
## ( inside: /..../aiconfigurator/src/aiconfigurator/systems/ )
_SYSTEM = "b200_sxm"
## String idenifier for the backend instance, used as the look up for perf database directory 
## (e.g: /scratch1/hanjiang/aiconfigurator/src/aiconfigurator/systems/data/b200_sxm/trtllm)
_BACKEND = "trtllm"
## The version is based on the performance database version. 
## ( /..../aiconfigurator/src/aiconfigurator/systems/data/b200_sxm/trtllm/1.2.0rc5 )
_VERSION = "1.2.0rc5"
## Three options: TRTLLMBackend, VLLMBackend, SGLANGBackend. Using TRTLLMBackend as default
_BACKEND_INSTANCE = TRTLLMBackend()




# ---------------------------------------------------------------------------
# Fixtures initialization for inputs of running different servings (agg, 
# disagg, disagg_afd)
# ---------------------------------------------------------------------------

@pytest.fixture
def perf_db():
    """Real PerfDatabase reading from actual system perf data."""
    systems_dir = get_system_config_path()
    return PerfDatabase(_SYSTEM, _BACKEND, _VERSION, systems_dir=str(systems_dir))


# ---------------------------------------------------------------------------
# The following has the runtime config like: parallelization strategy, 
# batch size, isl/osl for different test cases.
# ---------------------------------------------------------------------------

## This is also used as the default model config for agg mode 
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

# CTX_TOKENS is the batch size to go through one forwards pass. Similar to chunk size
_CTX_TOKENS = 6400

@pytest.fixture
def runtime_cfg():
    return RuntimeConfig(batch_size=2, isl=6400, osl=1000, beam_width=1)


@pytest.fixture
def disagg_session(perf_db):
    """Create a DisaggInferenceSession using the same DB for prefill and decode."""
    
    # Use the shared TensorRT LLM backend instance
    backend = _BACKEND_INSTANCE
    
    return DisaggInferenceSession(
        prefill_database=perf_db,
        prefill_backend=backend,
        decode_database=perf_db,
        decode_backend=backend,
    )

## AIConfigurator will use LRU for model info, placeholder and not used for now
@pytest.fixture(autouse=True)
def _clear_model_info_cache():
    """Clear the _get_model_info LRU cache between tests."""
    import aiconfigurator.sdk.models as models_module
    models_module._get_model_info.cache_clear()
    yield
    models_module._get_model_info.cache_clear()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAggBreakdown:



    def _run_agg(self, perf_db, runtime_cfg, model_config):
        """
        Build a fresh InferenceSession and call run_agg directly.
        Clears the backend cache first so database-mode switches take effect.
        Returns the InferenceSummary produced by run_agg.
        """
        backend = _BACKEND_INSTANCE
        backend._agg_cache.clear()
        model = get_model(_MODEL_PATH, model_config, backend.name.value)
        session = InferenceSession(model=model, database=perf_db, backend=backend)
        return session.run_agg(runtime_cfg, ctx_tokens=_CTX_TOKENS)

    def test_returns_all_time_breakdown(
        self, perf_db, runtime_cfg, prefill_model_config
    ):
        """Print a side-by-side SILICON / SOL / EMPIRICAL comparison of run_agg metrics."""
        result_silicon = self._run_agg(perf_db, runtime_cfg, prefill_model_config)

        perf_db.set_default_database_mode(DatabaseMode.SOL)
        result_sol = self._run_agg(perf_db, runtime_cfg, prefill_model_config)

        perf_db.set_default_database_mode(DatabaseMode.EMPIRICAL)
        result_emp = self._run_agg(perf_db, runtime_cfg, prefill_model_config)

        # Restore to SILICON so other tests are not affected
        perf_db.set_default_database_mode(DatabaseMode.SILICON)

        dict_si  = result_silicon.get_result_dict()
        dict_sol = result_sol.get_result_dict()
        dict_emp = result_emp.get_result_dict()

        print(f"\n{'=' * 110}")
        print(f"  AGG BREAKDOWN  (batch_size={runtime_cfg.batch_size}, isl={runtime_cfg.isl}, "
              f"osl={runtime_cfg.osl}, ctx_tokens={_CTX_TOKENS})")
        print(f"  {'metric':<34s}  {'SILICON':>14s}  {'SOL':>14s}  {'EMPIRICAL':>14s}  {'SI/SOL':>8s}  {'SI/EMP':>8s}")
        print(f"  {'-' * 104}")

        _PRINT_KEYS = [
            "request_rate", "bs", "global_bs", "ttft", "tpot",
            "seq/s", "seq/s/gpu", "tokens/s", "tokens/s/gpu",
            "tokens/s/user", "request_latency",
        ]

        def _fmt(v):
            return f"{v:14.4f}" if isinstance(v, (int, float)) else f"{str(v):>14s}"

        for key in _PRINT_KEYS:
            si  = dict_si[key]
            sol = dict_sol[key]
            emp = dict_emp[key]
            if isinstance(si, (int, float)) and isinstance(sol, (int, float)) and isinstance(emp, (int, float)):
                ratio_sol = f"{si / sol:8.2f}x" if sol > 0 else f"{'inf':>8s}"
                ratio_emp = f"{si / emp:8.2f}x" if emp > 0 else f"{'inf':>8s}"
            else:
                ratio_sol = ratio_emp = f"{'–':>8s}"
            print(f"  {key:<34s}  {_fmt(si)}  {_fmt(sol)}  {_fmt(emp)}  {ratio_sol}  {ratio_emp}")


class TestDisaggBreakdown:
    """Print a side-by-side SILICON / SOL / EMPIRICAL comparison of run_disagg metrics."""

    def _run_disagg(self, disagg_session, runtime_cfg, prefill_mc, decode_mc):
        """
        Call run_disagg and return the InferenceSummary.
        run_static has no caching, so no cache clearing is needed.
        """
        return disagg_session.run_disagg(
            model_path=_MODEL_PATH,
            runtime_config=runtime_cfg,
            prefill_model_config=prefill_mc,
            prefill_batch_size=runtime_cfg.batch_size,
            prefill_num_worker=1,
            decode_model_config=decode_mc,
            decode_batch_size=runtime_cfg.batch_size,
            decode_num_worker=1,
        )

    def test_returns_all_time_breakdown(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        """Print a side-by-side SILICON / SOL / EMPIRICAL comparison of run_disagg metrics."""
        result_silicon = self._run_disagg(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )

        prefill_db = disagg_session._prefill_database
        decode_db = disagg_session._decode_database

        prefill_db.set_default_database_mode(DatabaseMode.SOL)
        decode_db.set_default_database_mode(DatabaseMode.SOL)
        result_sol = self._run_disagg(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )

        prefill_db.set_default_database_mode(DatabaseMode.EMPIRICAL)
        decode_db.set_default_database_mode(DatabaseMode.EMPIRICAL)
        result_emp = self._run_disagg(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )

        # Restore to SILICON so other tests are not affected
        prefill_db.set_default_database_mode(DatabaseMode.SILICON)
        decode_db.set_default_database_mode(DatabaseMode.SILICON)

        dict_si  = result_silicon.get_result_dict()
        dict_sol = result_sol.get_result_dict()
        dict_emp = result_emp.get_result_dict()

        print(f"\n{'=' * 110}")
        print(f"  DISAGG BREAKDOWN  (batch_size={runtime_cfg.batch_size}, isl={runtime_cfg.isl}, "
              f"osl={runtime_cfg.osl})")
        print(f"  {'metric':<34s}  {'SILICON':>14s}  {'SOL':>14s}  {'EMPIRICAL':>14s}  {'SI/SOL':>8s}  {'SI/EMP':>8s}")
        print(f"  {'-' * 104}")

        _PRINT_KEYS = [
            "request_rate",
            "(p)bs", "(p)global_bs", "(p)workers",
            "(d)bs", "(d)global_bs", "(d)workers",
            "ttft", "tpot", "seq/s", "seq/s/gpu",
            "tokens/s", "tokens/s/gpu", "tokens/s/user", "request_latency",
        ]

        def _fmt(v):
            return f"{v:14.4f}" if isinstance(v, (int, float)) else f"{str(v):>14s}"

        for key in _PRINT_KEYS:
            si  = dict_si[key]
            sol = dict_sol[key]
            emp = dict_emp[key]
            if isinstance(si, (int, float)) and isinstance(sol, (int, float)) and isinstance(emp, (int, float)):
                ratio_sol = f"{si / sol:8.2f}x" if sol > 0 else f"{'inf':>8s}"
                ratio_emp = f"{si / emp:8.2f}x" if emp > 0 else f"{'inf':>8s}"
            else:
                ratio_sol = ratio_emp = f"{'–':>8s}"
            print(f"  {key:<34s}  {_fmt(si)}  {_fmt(sol)}  {_fmt(emp)}  {ratio_sol}  {ratio_emp}")


def _classify_op_stage(op_name: str) -> str:
    """Classify an op as 'attn' or 'ffn' using BaseModel's default classification."""
    # Use BaseModel's default _ATTN_COMPONENTS (MOEModel inherits the same set)
    name = op_name
    for pfx in ("context_", "generation_"):
        if name.startswith(pfx):
            name = name[len(pfx):]
            break
    return "attn" if name in BaseModel._ATTN_COMPONENTS else "ffn"


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
            prefill_batch_size=runtime_cfg.batch_size,
            prefill_num_worker=1,
            decode_model_config=decode_mc,
            decode_batch_size=runtime_cfg.batch_size,
            decode_num_worker=1,
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

