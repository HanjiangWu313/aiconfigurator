# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modular tests for Agg, Disagg, AFD breakdowns

Running cmd for breakdown tests (from the root of the aiconfigurator repo):

Agg:
python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestAggBreakdown::test_returns_all_time_breakdown -s

Disagg:
python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestDisaggBreakdown::test_returns_all_time_breakdown -s

Disagg AFD (session-level + per-kernel breakdown across attn:FFN GPU ratios):
python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestDisaggAFDBreakdown::test_returns_all_time_breakdown -s

MoEDispatch Runtime (diagnose why dispatch == 0 in single-GPU runs):
python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestMoEDispatchRuntime::test_dispatch_runtime_across_configs -s

"""

from pyexpat import model

import pytest

from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import DisaggInferenceSession, InferenceSession
from aiconfigurator.sdk.models import BaseModel, get_model
from aiconfigurator.sdk.operations import MoEDispatch
from aiconfigurator.sdk.perf_database import PerfDatabase, get_system_config_path
from aiconfigurator.sdk.common import DatabaseMode
pytestmark = pytest.mark.unit

## Follows the naming split with "/" like huggingface, list of models is in here: 
## /..../aiconfigurator/src/aiconfigurator/model_configs
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
    return PerfDatabase(_SYSTEM, _BACKEND, _VERSION, systems_root=str(systems_dir))


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


class TestAggAFDFFNLoadSignals:
    """Check that uneven AFD splits surface the expected FFN load signals in agg mode."""

    _AFD_CONFIGS = [
        (
            "4A:4F",
            ModelConfig(
                tp_size=1,
                pp_size=1,
                moe_tp_size=1,
                moe_ep_size=4,
                attention_dp_size=4,
                enable_afd=True,
            ),
        ),
        (
            "6A:2F",
            ModelConfig(
                tp_size=1,
                pp_size=1,
                moe_tp_size=1,
                moe_ep_size=2,
                attention_dp_size=6,
                enable_afd=True,
            ),
        ),
    ]

    @staticmethod
    def _run_agg(perf_db, runtime_cfg, model_config):
        backend = _BACKEND_INSTANCE
        backend._agg_cache.clear()
        model = get_model(_MODEL_PATH, model_config, backend.name.value)
        session = InferenceSession(model=model, database=perf_db, backend=backend)
        return session.run_agg(runtime_cfg, ctx_tokens=_CTX_TOKENS)

    def test_uneven_attn_ffn_split_increases_ffn_input_tokens(self, perf_db, runtime_cfg):
        results = {
            label: self._run_agg(perf_db, runtime_cfg, cfg).get_result_dict()
            for label, cfg in self._AFD_CONFIGS
        }

        even = results["4A:4F"]
        skewed = results["6A:2F"]

        print(f"\n{'=' * 110}")
        print("  AGG AFD FFN LOAD SIGNALS")
        print(f"  {'metric':<34s}  {'4A:4F':>14s}  {'6A:2F':>14s}  {'ratio':>10s}")
        print(f"  {'-' * 104}")
        for key in [
            "global_bs",
            "num_attn_gpus",
            "num_ffn_gpus",
            "ffn_mix_input_tokens",
            "ffn_mix_input_tokens_per_gpu",
            "ffn_gen_input_tokens",
            "ffn_gen_input_tokens_per_gpu",
            "tokens/s",
            "request_rate",
        ]:
            even_v = even[key]
            skewed_v = skewed[key]
            ratio = f"{skewed_v / even_v:10.2f}x" if even_v else f"{'inf':>10s}"
            if isinstance(even_v, (int, float)) and isinstance(skewed_v, (int, float)):
                print(f"  {key:<34s}  {even_v:14.4f}  {skewed_v:14.4f}  {ratio}")
            else:
                print(f"  {key:<34s}  {str(even_v):>14s}  {str(skewed_v):>14s}  {'-':>10s}")

        assert even["num_attn_gpus"] == 4
        assert even["num_ffn_gpus"] == 4
        assert skewed["num_attn_gpus"] == 6
        assert skewed["num_ffn_gpus"] == 2

        assert skewed["global_bs"] > even["global_bs"]
        assert skewed["ffn_mix_input_tokens"] > even["ffn_mix_input_tokens"]
        assert skewed["ffn_mix_input_tokens_per_gpu"] > even["ffn_mix_input_tokens_per_gpu"]
        assert skewed["ffn_gen_input_tokens"] > even["ffn_gen_input_tokens"]
        assert skewed["ffn_gen_input_tokens_per_gpu"] > even["ffn_gen_input_tokens_per_gpu"]


class TestDisaggBreakdown:
    """Print a side-by-side SILICON / SOL / EMPIRICAL comparison of run_disagg metrics."""

    def _run_disagg(self, disagg_session, runtime_cfg, prefill_mc, decode_mc):
        """
        Call run_disagg and return the InferenceSummary.
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


class TestMoEDispatchRuntime:
    """
    Diagnose and validate MoEDispatch communication latency across different
    parallelism configurations.

    Root cause of 0.0 in full test runs
    ------------------------------------
    ``MoEDispatch.query()`` returns ``comm_latency = 0`` whenever
    ``attention_tp_size == 1`` AND ``attention_dp_size == 1``.
    This is the single-GPU path: with the default
    ``moe_tp_size=1, moe_ep_size=1, attention_dp_size=1`` there is no
    inter-GPU dispatch communication, so the latency is correctly 0.

    To get non-zero dispatch latency you need either:
      * ``attention_dp_size > 1``  → drives all2all / nvfp4-all2all
      * OR ``attention_tp_size > 1`` (= ``tp_size`` when ``attention_dp_size=1``)
                                  → drives allreduce

    where ``attention_tp_size = moe_tp_size * moe_ep_size / attention_dp_size``.
    Both must satisfy the MOEModel constraint:
        ``tp_size * attention_dp_size == moe_tp_size * moe_ep_size``
    """

    # -----------------------------------------------------------------
    # (label, ModelConfig kwargs) sweep table
    # -----------------------------------------------------------------
    _CONFIGS = [
        (
            "baseline  ep=1  attn_dp=1",
            dict(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=1, attention_dp_size=1),
        ),
        (
            "ep=4  attn_dp=4",
            dict(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=4, attention_dp_size=4),
        ),
        (
            "ep=8  attn_dp=8",
            dict(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=8, attention_dp_size=8),
        ),
        (
            "tp=4  moe_tp=4  ep=1  attn_dp=1",
            dict(tp_size=4, pp_size=1, moe_tp_size=4, moe_ep_size=1, attention_dp_size=1),
        ),
    ]

    # The four dispatch op names produced by MOEModel
    _DISPATCH_OPS = [
        "context_moe_pre_dispatch",
        "context_moe_post_dispatch",
        "generation_moe_pre_dispatch",
        "generation_moe_post_dispatch",
    ]

    def _run_ctx_gen_static(
        self, perf_db, model_config
    ) -> tuple[dict, dict]:
        """
        Build model + InferenceSession, run static_ctx and static_gen,
        return (context_latency_dict, generation_latency_dict).
        Cache is cleared so database-mode switches take effect.
        """
        _BACKEND_INSTANCE._agg_cache.clear()
        model = get_model(_MODEL_PATH, model_config, _BACKEND_INSTANCE.name.value)
        session = InferenceSession(model=model, database=perf_db, backend=_BACKEND_INSTANCE)
        ctx_summary = session.run_static(
            RuntimeConfig(batch_size=2, isl=6400, osl=1000, beam_width=1), mode="static_ctx"
        )
        gen_summary = session.run_static(
            RuntimeConfig(batch_size=2, isl=6400, osl=1000, beam_width=1), mode="static_gen"
        )
        return ctx_summary.get_context_latency_dict(), gen_summary.get_generation_latency_dict(), model

    def test_dispatch_runtime_across_configs(self, perf_db):
        """
        Print MoEDispatch dispatch-op latency (ms) for every config in ``_CONFIGS``,
        across all three database modes side-by-side:

        Database modes: Silicon, SOL, EMPIRICAL
        Expected observations
        ---------------------
        * ``baseline ep=1 attn_dp=1``   : all dispatch ops == 0.0 ms  (single GPU, no comms)
        * ``ep=4  attn_dp=4``           : dispatch > 0  (nvfp4-all2all path on SM100/B200)
        * ``ep=8  attn_dp=8``           : dispatch > 0  (larger all2all volume)
        * ``tp=4 moe_tp=4 ep=1 dp=1``  : dispatch > 0  (allreduce path, attention_tp_size=4)

        Running cmd:
            python -m pytest tests/unit/sdk/models/test_agg_disagg_afd.py::TestMoEDispatchRuntime::test_dispatch_runtime_across_configs -s
        """
        _MODES = [
            ("SILICON",   DatabaseMode.SILICON),
            ("SOL",       DatabaseMode.SOL),
            ("EMPIRICAL", DatabaseMode.EMPIRICAL),
        ]

        # Collect results for all configs × all modes.
        # all_results[mode_label][config_label] = (ctx_d, gen_d)
        all_results: dict[str, dict[str, tuple]] = {}
        models_by_label: dict[str, object] = {}

        for mode_label, db_mode in _MODES:
            perf_db.set_default_database_mode(db_mode)
            all_results[mode_label] = {}
            for label, cfg_kwargs in self._CONFIGS:
                mc = ModelConfig(**cfg_kwargs)
                ctx_d, gen_d, mdl = self._run_ctx_gen_static(perf_db, mc)
                all_results[mode_label][label] = (ctx_d, gen_d)
                if mode_label == "SILICON":
                    models_by_label[label] = mdl

        # Restore to SILICON so other tests are not affected
        perf_db.set_default_database_mode(DatabaseMode.SILICON)

        labels = [label for label, _ in self._CONFIGS]

        # ---- Table 0: NCCL primitive names from MoEDispatch object attributes ----
        # The latency dict keys are model-level op names (same for every config).
        # The actual NCCL primitive is determined by the op's computed attributes:
        #   _attention_tp_size, _attention_dp_size, _enable_fp4_all2all, _pre_dispatch
        def _primitive(op: MoEDispatch, sm_ver: int) -> str:
            if op._attention_tp_size > 1:
                return "custom_allreduce"
            elif op._attention_dp_size > 1:
                if sm_ver == 100 and op._enable_fp4_all2all:
                    # pre: 2× alltoall (main + scale-factor) + 10µs startup
                    # post: 1× alltoall
                    return "nvfp4-alltoall ×2" if op._pre_dispatch else "nvfp4-alltoall ×1"
                else:
                    return "all_gather" if op._pre_dispatch else "reduce_scatter"
            else:
                return "— (no comm)"

        sm_version = perf_db.system_spec["gpu"]["sm_version"]
        prim_col_w = 22
        print(f"\n{'=' * 130}")
        print("  MOE DISPATCH NCCL PRIMITIVES — read from MoEDispatch op object attributes")
        print(f"  (attn_tp = moe_tp × moe_ep / attn_dp;  SM{sm_version})")
        print(f"\n  {'config':<36s}  {'attn_tp':>8s}  {'attn_dp':>8s}  {'ctx pre_dispatch':<{prim_col_w}s}  {'ctx post_dispatch':<{prim_col_w}s}  {'gen pre_dispatch':<{prim_col_w}s}  {'gen post_dispatch':<{prim_col_w}s}")
        print(f"  {'-' * 130}")
        for label in labels:
            mdl = models_by_label[label]
            ctx_dispatch = {op._name: op for op in mdl.context_ops if isinstance(op, MoEDispatch)}
            gen_dispatch  = {op._name: op for op in mdl.generation_ops if isinstance(op, MoEDispatch)}
            ctx_pre  = next((op for op in ctx_dispatch.values() if op._pre_dispatch), None)
            ctx_post = next((op for op in ctx_dispatch.values() if not op._pre_dispatch), None)
            gen_pre  = next((op for op in gen_dispatch.values() if op._pre_dispatch), None)
            gen_post = next((op for op in gen_dispatch.values() if not op._pre_dispatch), None)
            attn_tp = ctx_pre._attention_tp_size if ctx_pre else 1
            attn_dp = ctx_pre._attention_dp_size if ctx_pre else 1
            p_ctx_pre  = _primitive(ctx_pre,  sm_version) if ctx_pre  else "—"
            p_ctx_post = _primitive(ctx_post, sm_version) if ctx_post else "—"
            p_gen_pre  = _primitive(gen_pre,  sm_version) if gen_pre  else "—"
            p_gen_post = _primitive(gen_post, sm_version) if gen_post else "—"
            print(f"  {label:<36s}  {attn_tp:>8d}  {attn_dp:>8d}  {p_ctx_pre:<{prim_col_w}s}  {p_ctx_post:<{prim_col_w}s}  {p_gen_pre:<{prim_col_w}s}  {p_gen_post:<{prim_col_w}s}")

        # ---- Per-config tables: dispatch ops × SILICON / SOL / EMPIRICAL ----
        # Mirrors the style of TestDisaggAFDBreakdown.test_returns_all_time_breakdown.
        for label in labels:
            si_ctx,  si_gen  = all_results["SILICON"][label]
            sol_ctx, sol_gen = all_results["SOL"][label]
            emp_ctx, emp_gen = all_results["EMPIRICAL"][label]

            for phase_label, ops_si, ops_sol, ops_emp in [
                ("CTX (prefill)", si_ctx,  sol_ctx, emp_ctx),
                ("GEN (decode)",  si_gen,  sol_gen, emp_gen),
            ]:
                dispatch_ops = [k for k in ops_si if "dispatch" in k]
                total_si  = sum(ops_si.values())
                total_sol = sum(ops_sol.values())
                total_emp = sum(ops_emp.values())
                disp_si   = sum(ops_si[k]  for k in dispatch_ops)
                disp_sol  = sum(ops_sol.get(k, 0.0) for k in dispatch_ops)
                disp_emp  = sum(ops_emp.get(k, 0.0) for k in dispatch_ops)

                print(f"\n{'=' * 110}")
                print(f"  [{label}]  {phase_label}")
                print(f"  dispatch total:  SILICON={disp_si:.4f} ms,  SOL={disp_sol:.4f} ms,  EMPIRICAL={disp_emp:.4f} ms")
                print(f"  {'op':<44s}  {'SILICON':>10s}  {'SOL':>10s}  {'EMPIRICAL':>10s}  {'SI/SOL':>8s}  {'SI/EMP':>8s}")
                print(f"  {'-' * 104}")
                for op in dispatch_ops:
                    si  = ops_si.get(op, 0.0)
                    sol = ops_sol.get(op, 0.0)
                    emp = ops_emp.get(op, 0.0)
                    ratio_sol = f"{si / sol:8.2f}x" if sol > 0 else f"{'—':>8s}"
                    ratio_emp = f"{si / emp:8.2f}x" if emp > 0 else f"{'—':>8s}"
                    print(f"  {op:<44s}  {si:10.4f}  {sol:10.4f}  {emp:10.4f}  {ratio_sol}  {ratio_emp}")
                # Dispatch fraction row
                pct_si  = 100.0 * disp_si  / total_si  if total_si  > 0 else 0.0
                pct_sol = 100.0 * disp_sol / total_sol if total_sol > 0 else 0.0
                pct_emp = 100.0 * disp_emp / total_emp if total_emp > 0 else 0.0
                print(f"  {'-' * 104}")
                print(f"  {'DISPATCH TOTAL':<44s}  {disp_si:10.4f}  {disp_sol:10.4f}  {disp_emp:10.4f}")
                print(f"  {'DISPATCH / PHASE TOTAL':<44s}  {pct_si:9.2f}%  {pct_sol:9.2f}%  {pct_emp:9.2f}%")

        # ---- Assertions (use SILICON results) ----
        # 1. Baseline (single GPU): all dispatch ops must be exactly 0
        base_label = self._CONFIGS[0][0]
        si_ctx_base, si_gen_base = all_results["SILICON"][base_label]
        for op in self._DISPATCH_OPS:
            d = si_ctx_base if op.startswith("context_") else si_gen_base
            v = d.get(op, 0.0)
            assert v == 0.0, (
                f"Expected dispatch=0 for baseline single-GPU config, got {v:.6f} for '{op}'"
            )

        # 2. Multi-GPU configs: at least one dispatch op must be non-zero (SILICON)
        for label, _ in self._CONFIGS[1:]:
            si_ctx, si_gen = all_results["SILICON"][label]
            all_vals = [
                (si_ctx if op.startswith("context_") else si_gen).get(op, 0.0)
                for op in self._DISPATCH_OPS
            ]
            assert any(v > 0 for v in all_vals), (
                f"Expected non-zero MoEDispatch latency for config '{label}', "
                f"but all dispatch ops returned 0: {dict(zip(self._DISPATCH_OPS, all_vals))}"
            )


def _classify_op_stage(op_name: str) -> str:
    """Classify an op as 'attn' or 'ffn' using BaseModel's substring classifier.

    Single source of truth: delegate to ``BaseModel._is_attn_op`` so the test
    classification stays in sync with the backend's own AFD partitioning.
    """
    _bm = BaseModel.__new__(BaseModel)
    return "attn" if _bm._is_attn_op(op_name) else "ffn"


class TestDisaggAFDBreakdown:
    """
    Verify that ``run_disagg_afd`` produces correct attention / FFN latency
    breakdowns for both prefill and decode workers.

    Also includes per-kernel AFD runtime breakdown across different
    attention : FFN GPU allocation ratios.  The backend query loop now
    forwards ``num_attn_gpus`` / ``num_ffn_gpus`` from ``ModelConfig``
    to :meth:`MoEDispatch.query` automatically.  The per-kernel tests
    iterate model ops directly and inject the AFD-specific kwargs to
    mirror this behavior.
    """

    # (label, ModelConfig kwargs, num_attn_gpus, num_ffn_gpus)
    _AFD_CONFIGS = [
        (
            "ep8  4A:4F",
            dict(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=8,
                 attention_dp_size=8, enable_afd=True,
                 num_attn_gpus=4, num_ffn_gpus=4),
            4, 4,
        ),
        (
            "ep8  2A:6F",
            dict(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=8,
                 attention_dp_size=8, enable_afd=True,
                 num_attn_gpus=2, num_ffn_gpus=6),
            2, 6,
        ),
        (
            "ep8  6A:2F",
            dict(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=8,
                 attention_dp_size=8, enable_afd=True,
                 num_attn_gpus=6, num_ffn_gpus=2),
            6, 2,
        ),
        (
            "ep8  1A:7F",
            dict(tp_size=1, pp_size=1, moe_tp_size=1, moe_ep_size=8,
                 attention_dp_size=8, enable_afd=True,
                 num_attn_gpus=1, num_ffn_gpus=7),
            1, 7,
        ),
    ]

    _BATCH_SIZE = 2
    _ISL = 6400

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_ctx_ops(ops, database, batch_size, isl, num_attn_gpus, num_ffn_gpus):
        """Query context-phase ops, injecting AFD kwargs for MoEDispatch."""
        latency: dict[str, float] = {}
        for op in ops:
            x = batch_size * isl if "logits_gemm" not in op._name else batch_size
            kwargs = dict(x=x, batch_size=batch_size, beam_width=1, s=isl, prefix=0)
            if isinstance(op, MoEDispatch) and op._enable_afd:
                kwargs["num_attn_gpus"] = num_attn_gpus
                kwargs["num_ffn_gpus"] = num_ffn_gpus
            result = op.query(database, **kwargs)
            latency[op._name] = latency.get(op._name, 0.0) + float(result)
        return latency

    @staticmethod
    def _query_gen_ops(ops, database, batch_size, isl, num_attn_gpus, num_ffn_gpus):
        """Query a single generation step, injecting AFD kwargs for MoEDispatch."""
        latency: dict[str, float] = {}
        for op in ops:
            x = batch_size
            kwargs = dict(x=x, batch_size=batch_size, beam_width=1, s=isl + 1)
            if isinstance(op, MoEDispatch) and op._enable_afd:
                kwargs["num_attn_gpus"] = num_attn_gpus
                kwargs["num_ffn_gpus"] = num_ffn_gpus
            result = op.query(database, **kwargs)
            latency[op._name] = latency.get(op._name, 0.0) + float(result)
        return latency

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
        self, perf_db, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):  
        """run_disagg_afd should return a dict with all documented keys.

        Part 1 — Session-level breakdown:
            Prints a side-by-side comparison of SILICON / SOL / EMPIRICAL
            latencies for prefill-attn, prefill-ffn, decode-attn, decode-ffn.

        Part 2 — Per-kernel AFD breakdown:
            Iterates model ops directly across 4 attn:FFN GPU ratios,
            printing per-kernel latency tables with stage classification
            and dispatch P2P cross-config comparison.
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

        # ==================================================================
        # Part 2: Per-kernel AFD breakdown across attn:FFN GPU ratios
        # ==================================================================
        _MODES = [
            ("SILICON",   DatabaseMode.SILICON),
            ("SOL",       DatabaseMode.SOL),
            ("EMPIRICAL", DatabaseMode.EMPIRICAL),
        ]

        # all_afd[mode_label][config_label] = (ctx_dict, gen_dict)
        all_afd: dict[str, dict[str, tuple]] = {}

        for mode_label, db_mode in _MODES:
            perf_db.set_default_database_mode(db_mode)
            all_afd[mode_label] = {}
            for label, cfg_kwargs, n_attn, n_ffn in self._AFD_CONFIGS:
                mc = ModelConfig(**cfg_kwargs)
                _BACKEND_INSTANCE._agg_cache.clear()
                model = get_model(_MODEL_PATH, mc, _BACKEND_INSTANCE.name.value)
                ctx_d = self._query_ctx_ops(
                    model.context_ops, perf_db, self._BATCH_SIZE, self._ISL,
                    n_attn, n_ffn,
                )
                gen_d = self._query_gen_ops(
                    model.generation_ops, perf_db, self._BATCH_SIZE, self._ISL,
                    n_attn, n_ffn,
                )
                all_afd[mode_label][label] = (ctx_d, gen_d)

        perf_db.set_default_database_mode(DatabaseMode.SILICON)

        # ---- per-config, per-phase table ----
        for label, cfg_kwargs, n_attn, n_ffn in self._AFD_CONFIGS:
            for phase_idx, phase_label in [(0, "CTX (prefill)"), (1, "GEN (decode)")]:
                ops_si  = all_afd["SILICON"][label][phase_idx]
                ops_sol = all_afd["SOL"][label][phase_idx]
                ops_emp = all_afd["EMPIRICAL"][label][phase_idx]

                total_si  = sum(ops_si.values())
                total_sol = sum(ops_sol.values())
                total_emp = sum(ops_emp.values())

                dispatch_keys = [k for k in ops_si if "dispatch" in k]
                dispatch_si   = sum(ops_si[k] for k in dispatch_keys)
                attn_keys = [k for k in ops_si if _classify_op_stage(k) == "attn"]
                ffn_keys  = [k for k in ops_si if _classify_op_stage(k) == "ffn"]
                attn_si = sum(ops_si[k] for k in attn_keys)
                ffn_si  = sum(ops_si[k] for k in ffn_keys)

                print(f"\n{'=' * 120}")
                print(f"  [{label}]  {phase_label}  (Attn GPUs={n_attn}, FFN GPUs={n_ffn})")
                print(f"  TOTAL: SI={total_si:.4f}ms  SOL={total_sol:.4f}ms  EMP={total_emp:.4f}ms")
                if total_si > 0:
                    print(
                        f"  Attn: {attn_si:.4f}ms ({100 * attn_si / total_si:.1f}%)   "
                        f"FFN: {ffn_si:.4f}ms ({100 * ffn_si / total_si:.1f}%)   "
                        f"Dispatch P2P: {dispatch_si:.4f}ms ({100 * dispatch_si / total_si:.1f}%)"
                    )
                print(
                    f"  {'kernel':<42s} {'stage':>5s}  {'SILICON':>10s}  {'SOL':>10s}"
                    f"  {'EMP':>10s}  {'SI/SOL':>8s}  {'SI/EMP':>8s}"
                )
                print(f"  {'-' * 118}")

                for op in ops_si:
                    si  = ops_si.get(op, 0.0)
                    sol = ops_sol.get(op, 0.0)
                    emp = ops_emp.get(op, 0.0)
                    stage = _classify_op_stage(op)
                    r_sol = f"{si / sol:8.2f}x" if sol > 0 else f"{'—':>8s}"
                    r_emp = f"{si / emp:8.2f}x" if emp > 0 else f"{'—':>8s}"
                    tag = " ◄P2P" if "dispatch" in op else ""
                    print(
                        f"  {op + tag:<42s} {stage:>5s}  {si:10.4f}  {sol:10.4f}"
                        f"  {emp:10.4f}  {r_sol}  {r_emp}"
                    )

                print(f"  {'-' * 118}")
                disp_sol = sum(ops_sol.get(k, 0) for k in dispatch_keys)
                disp_emp = sum(ops_emp.get(k, 0) for k in dispatch_keys)
                print(
                    f"  {'DISPATCH P2P TOTAL':<42s} {'':>5s}  {dispatch_si:10.4f}"
                    f"  {disp_sol:10.4f}  {disp_emp:10.4f}"
                )
                pct_si  = 100.0 * dispatch_si / total_si  if total_si  > 0 else 0.0
                pct_sol = 100.0 * disp_sol    / total_sol if total_sol > 0 else 0.0
                pct_emp = 100.0 * disp_emp    / total_emp if total_emp > 0 else 0.0
                print(
                    f"  {'DISPATCH P2P / TOTAL':<42s} {'':>5s}  {pct_si:9.2f}%"
                    f"  {pct_sol:9.2f}%  {pct_emp:9.2f}%"
                )

        # ---- Cross-config dispatch comparison (SILICON) ----
        print(f"\n{'=' * 120}")
        print("  AFD DISPATCH P2P COMPARISON ACROSS CONFIGS (SILICON)")
        print(
            f"  {'config':<20s}  {'ctx_pre':>10s}  {'ctx_post':>10s}"
            f"  {'gen_pre':>10s}  {'gen_post':>10s}"
            f"  {'ctx_disp':>10s}  {'gen_disp':>10s}"
        )
        print(f"  {'-' * 100}")
        for label, _, n_attn, n_ffn in self._AFD_CONFIGS:
            ctx_d, gen_d = all_afd["SILICON"][label]
            ctx_pre  = ctx_d.get("context_moe_pre_dispatch", 0.0)
            ctx_post = ctx_d.get("context_moe_post_dispatch", 0.0)
            gen_pre  = gen_d.get("generation_moe_pre_dispatch", 0.0)
            gen_post = gen_d.get("generation_moe_post_dispatch", 0.0)
            print(
                f"  {label:<20s}  {ctx_pre:10.4f}  {ctx_post:10.4f}"
                f"  {gen_pre:10.4f}  {gen_post:10.4f}"
                f"  {ctx_pre + ctx_post:10.4f}  {gen_pre + gen_post:10.4f}"
            )

        # ---- Per-kernel assertions ----
        for label, _, _, _ in self._AFD_CONFIGS:
            ctx_d, gen_d = all_afd["SILICON"][label]
            for d in (ctx_d, gen_d):
                for op_name, val in d.items():
                    if "dispatch" in op_name:
                        assert val > 0, (
                            f"Expected {op_name} > 0 for config '{label}', "
                            f"got {val:.6f}"
                        )

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

    def test_afd_enabled_with_astrasim(
        self, disagg_session, runtime_cfg, prefill_model_config, decode_model_config
    ):
        result = self._run_afd(
            disagg_session, runtime_cfg, prefill_model_config, decode_model_config
        )
        assert "prefill_attn" in result
    
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

    # ---- per-kernel AFD dispatch assertions ----

    def test_dispatch_pre_gt_post(self, perf_db):
        """
        Pre-dispatch (attn → FFN) includes routing fanout (k_avg > 1);
        post-dispatch (FFN → attn) does not.  For Qwen3-30B-A3B (topk=8)
        the pre-dispatch latency should always exceed post-dispatch.
        """
        perf_db.set_default_database_mode(DatabaseMode.SILICON)
        for label, cfg_kwargs, n_attn, n_ffn in self._AFD_CONFIGS:
            mc = ModelConfig(**cfg_kwargs)
            _BACKEND_INSTANCE._agg_cache.clear()
            model = get_model(_MODEL_PATH, mc, _BACKEND_INSTANCE.name.value)

            ctx_d = self._query_ctx_ops(
                model.context_ops, perf_db, self._BATCH_SIZE, self._ISL,
                n_attn, n_ffn,
            )
            pre  = ctx_d.get("context_moe_pre_dispatch", 0.0)
            post = ctx_d.get("context_moe_post_dispatch", 0.0)
            assert pre > post, (
                f"[{label}] Expected pre-dispatch ({pre:.4f} ms) > "
                f"post-dispatch ({post:.4f} ms) due to routing fanout"
            )

    def test_non_dispatch_ops_invariant(self, perf_db):
        """
        Non-dispatch kernel latencies should be identical across AFD configs
        (only the dispatch P2P changes with attn:FFN ratio).
        """
        perf_db.set_default_database_mode(DatabaseMode.SILICON)

        ref_ctx: dict | None = None
        ref_gen: dict | None = None
        ref_label: str = ""

        for label, cfg_kwargs, n_attn, n_ffn in self._AFD_CONFIGS:
            mc = ModelConfig(**cfg_kwargs)
            _BACKEND_INSTANCE._agg_cache.clear()
            model = get_model(_MODEL_PATH, mc, _BACKEND_INSTANCE.name.value)

            ctx_d = self._query_ctx_ops(
                model.context_ops, perf_db, self._BATCH_SIZE, self._ISL,
                n_attn, n_ffn,
            )
            gen_d = self._query_gen_ops(
                model.generation_ops, perf_db, self._BATCH_SIZE, self._ISL,
                n_attn, n_ffn,
            )

            # Keep only non-dispatch ops
            ctx_nd = {k: v for k, v in ctx_d.items() if "dispatch" not in k}
            gen_nd = {k: v for k, v in gen_d.items() if "dispatch" not in k}

            if ref_ctx is None:
                ref_ctx, ref_gen, ref_label = ctx_nd, gen_nd, label
                continue

            for op, val in ctx_nd.items():
                ref_val = ref_ctx.get(op, 0.0)
                assert abs(val - ref_val) < 1e-6, (
                    f"CTX op '{op}' differs between '{ref_label}' ({ref_val:.6f}) "
                    f"and '{label}' ({val:.6f})"
                )
            for op, val in gen_nd.items():
                ref_val = ref_gen.get(op, 0.0)
                assert abs(val - ref_val) < 1e-6, (
                    f"GEN op '{op}' differs between '{ref_label}' ({ref_val:.6f}) "
                    f"and '{label}' ({val:.6f})"
                )
