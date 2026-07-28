<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AFD and AstraSim validation

This workflow compares the AFD attention, FFN, and attention-to-FFN transfer
breakdowns produced by AI Configurator with AstraSim enabled and disabled.

The comparison has two distinct data sources:

- Attention and FFN compute latencies come from the selected AI Configurator
  performance database.
- AFD transfer latency comes from AstraSim when `--use-astrasim` is set. With
  `--no-use-astrasim`, AI Configurator uses its analytical or silicon
  communication fallback.

The repository does not contain measured AFD network traces from a deployed
system. Therefore, the two runs validate model integration and quantify the
network-model delta; external measurements are still required for
AstraSim-versus-hardware accuracy validation.

## Set up a fresh clone

Git LFS is required because the performance tables are stored as LFS objects.
The AFD work is published on the `change-with-upstream-merge` branch, so select
that branch explicitly when cloning the fork.

```bash
git clone --branch change-with-upstream-merge <fork-url>
cd aiconfigurator
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The AstraSim Python extensions are native build products and are intentionally
not committed. Build them with the same Python interpreter that will run AI
Configurator:

```bash
python -m pip install pybind11
cmake \
  -S network_backend/astra-network-analytical \
  -B network_backend/astra-network-analytical/build
cmake --build network_backend/astra-network-analytical/build -j
```

Verify that both bindings load:

```bash
python - <<'PY'
from aiconfigurator.sdk import astrasim_utils

assert astrasim_utils.NETWORK_SIM_AVAILABLE, "congestion-aware AstraSim binding is unavailable"
assert astrasim_utils.NETWORK_SIM_UNAWARE_AVAILABLE, "congestion-unaware AstraSim binding is unavailable"
print("Both AstraSim bindings are available.")
PY
```

If either assertion fails, confirm that the active Python version matches the
extension suffix under `network_backend/astra-network-analytical/lib/`, then
reconfigure and rebuild.

## Run focused regression checks

```bash
pytest \
  tests/unit/sdk/backends/test_base_backend.py \
  tests/unit/sdk/models/test_moe_afd_ops.py \
  tests/unit/sdk/test_kv_cache_transfer_layout.py
python tools/test_astrasim_afd_transfer.py
python tools/test_multi_tier_topology.py
python tools/test_afd_pipeline.py -m 4 -b 1 --tp 1 --dp 1 --moe-tp 1 --moe-ep 4
```

## Compare network models

Run the same configuration twice and change only the network-model flag:

```bash
python tools/afd_transfer_tracker.py \
  --serving agg \
  --isls 6400 \
  --osls 1000 \
  --tp 1 \
  --dp 4 \
  --moe-tp 1 \
  --moe-ep 4 \
  --database-mode SILICON \
  --use-astrasim \
  --csv /tmp/afd_astrasim.csv

python tools/afd_transfer_tracker.py \
  --serving agg \
  --isls 6400 \
  --osls 1000 \
  --tp 1 \
  --dp 4 \
  --moe-tp 1 \
  --moe-ep 4 \
  --database-mode SILICON \
  --no-use-astrasim \
  --csv /tmp/afd_analytical.csv
```

Compare `attn_ms`, `ffn_ms`, `a2f_ms`, `f2a_ms`, `afd_transfer_ms`, and their
percentage columns between the two CSV files. The attention and FFN compute
inputs should remain the same; the AFD communication model is the controlled
difference. The tracker exits immediately if `--use-astrasim` is requested but
either native binding is unavailable.
