"""
AstraSim network simulator utilities.

Provides :class:`AstraSimManager` – a single entry-point for every
network-simulation need (P2P, collectives, KV-cache transfer).

Topology YAML files are generated **on-the-fly** from system-spec
parameters and cached on disk so duplicate files are never created.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

# Default network config path (kept for backward-compat)
DEFAULT_NETWORK_CONFIG = os.path.join(
    _PROJECT_ROOT, "network_backend", "astra-network-analytical", "input", "Ring.yml"
)

# Network simulator library path
_NETWORK_SIM_LIB_PATH = os.path.join(
    _PROJECT_ROOT, "network_backend", "astra-network-analytical", "lib"
)

# Default directory for auto-generated topology YAML files
_DEFAULT_TOPOLOGY_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "network_backend", "astra-network-analytical", "input", "auto_generated"
)

# ---------------------------------------------------------------------------
# AstraSim C++ binding imports (best-effort)
# ---------------------------------------------------------------------------
NETWORK_SIM_AVAILABLE = False
EventQueue = None
Topology = None
Chunk = None
NetworkParser = None
construct_topology = None

if _NETWORK_SIM_LIB_PATH not in sys.path:
    sys.path.insert(0, _NETWORK_SIM_LIB_PATH)

try:
    from simulation_py_congestion_aware import (
        EventQueue,
        Topology,
        Chunk,
        NetworkParser,
        construct_topology,
    )
    NETWORK_SIM_AVAILABLE = True
    logger.info(f"AstraSim network simulator loaded from {_NETWORK_SIM_LIB_PATH}")
except ImportError as e:
    NETWORK_SIM_AVAILABLE = False
    logger.warning(f"AstraSim network simulator not available: {e}")
    logger.warning("Network latency modeling will be disabled")


# ---------------------------------------------------------------------------
# Helpers for checking availability and default config
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """Check if AstraSim is available."""
    return NETWORK_SIM_AVAILABLE


def get_default_config() -> str:
    """Get default network configuration file path."""
    return DEFAULT_NETWORK_CONFIG


# ---------------------------------------------------------------------------
# Topology YAML generation & caching
# ---------------------------------------------------------------------------
_VALID_TOPOLOGIES = ("Ring", "Switch", "FullyConnected")


def _topology_filename(
    topology: str, npus_count: int, bandwidth_gbps: float, latency_ns: float
) -> str:
    """Deterministic filename for a given set of topology parameters."""
    # Use a short hash to keep names readable but collision-free
    key = f"{topology}_{npus_count}_{bandwidth_gbps}_{latency_ns}"
    short_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"auto_{topology}_{npus_count}npus_{bandwidth_gbps}gbps_{short_hash}.yml"


def get_or_create_topology_config(
    npus_count: int,
    bandwidth_gbps: float,
    latency_ns: float = 500.0,
    topology: str = "Ring",
    cache_dir: str | None = None,
) -> str:
    """Return path to a topology YAML file, creating it on-the-fly if needed.

    The file is written into *cache_dir* (default:
    ``network_backend/…/input/auto_generated/``) and is reused on
    subsequent calls with identical parameters.

    Args:
        npus_count: Number of NPUs (GPUs) in the topology.
        bandwidth_gbps: Link bandwidth in **GB/s**.
        latency_ns: Link latency in **nanoseconds** (default 500).
        topology: One of ``"Ring"``, ``"Switch"``, ``"FullyConnected"``.
        cache_dir: Directory for cached topology files.

    Returns:
        Absolute path to the (possibly newly created) YAML file.
    """
    if topology not in _VALID_TOPOLOGIES:
        raise ValueError(
            f"Invalid topology '{topology}'. Must be one of {_VALID_TOPOLOGIES}"
        )

    cache_dir = cache_dir or _DEFAULT_TOPOLOGY_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    filename = _topology_filename(topology, npus_count, bandwidth_gbps, latency_ns)
    filepath = os.path.join(cache_dir, filename)

    if os.path.isfile(filepath):
        logger.debug(f"Reusing cached topology config: {filepath}")
        return filepath

    content = (
        f"# Auto-generated AstraSim topology config\n"
        f"topology: [ {topology} ]\n"
        f"npus_count: [ {npus_count} ]\n"
        f"bandwidth: [ {bandwidth_gbps} ]  # GB/s\n"
        f"latency: [ {latency_ns} ]  # ns\n"
    )

    # Atomic write via temp file + rename to avoid partial reads
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".yml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, filepath)
    except Exception:
        # Clean up on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(f"Created topology config: {filepath}")
    return filepath


def derive_network_params_from_system_spec(
    system_spec: dict,
    num_gpus: int,
    topology: str | None = None,
) -> dict:
    """Derive AstraSim topology parameters from a system YAML spec.

    Supports up to **three bandwidth tiers** to accommodate different
    system architectures:

    * **Tier 1 – intra-node** (``intra_node_bw``):
      NVLink within a single physical node.
      Applies when ``num_gpus <= num_gpus_per_node``.
    * **Tier 2 – inter-node / intra-rack** (``inter_node_bw``):
      NVLink via NVSwitch within a rack (e.g. GB200 NVL72) or
      InfiniBand between nodes in a 2-tier system (e.g. H100, B200).
      Applies when ``num_gpus <= num_gpus_per_rack`` (or when there
      is no ``num_gpus_per_rack`` key and ``num_gpus > num_gpus_per_node``).
    * **Tier 3 – inter-rack** (``inter_rack_bw``):
      InfiniBand between racks.  Only present on systems like GB200
      that define ``num_gpus_per_rack`` and ``inter_rack_bw``.
      Applies when ``num_gpus > num_gpus_per_rack``.

    Example tier selection for each system:

    * **H100 SXM** (NVL8, 2-tier):
      ≤ 8 GPUs → 450 GB/s NVLink · > 8 GPUs → 25 GB/s IB
    * **B200 SXM** (NVL8, 2-tier):
      ≤ 8 GPUs → 900 GB/s NVLink · > 8 GPUs → 50 GB/s IB
    * **GB200 SXM** (NVL4 + NVL72 rack, 3-tier):
      ≤ 4 GPUs → 900 GB/s NVLink (intra-node)
      ≤ 72 GPUs → 900 GB/s NVLink via NVSwitch (intra-rack)
      > 72 GPUs → 25 GB/s IB (inter-rack)

    Args:
        system_spec: Parsed system YAML dict (must contain ``node`` key).
        num_gpus: Total number of GPUs in the topology.
        topology: Override topology type.  ``None`` → auto-select
            (``FullyConnected`` for intra-node, ``Ring`` / ``Switch``
            for inter-node / inter-rack).

    Returns:
        dict with keys ``npus_count``, ``bandwidth_gbps``, ``latency_ns``,
        ``topology``, ``tier``.
    """
    node = system_spec["node"]
    num_gpus_per_node = node["num_gpus_per_node"]
    num_gpus_per_rack = node.get("num_gpus_per_rack")  # None for 2-tier systems

    if num_gpus <= num_gpus_per_node:
        # ── Tier 1: intra-node (NVLink) ──────────────────────────
        bw_bytes_per_s = node["intra_node_bw"]
        latency_s = node.get("p2p_latency", 0.0)
        auto_topology = "FullyConnected"
        tier = "intra-node"

    elif num_gpus_per_rack is not None and num_gpus <= num_gpus_per_rack:
        # ── Tier 2: inter-node / intra-rack (NVSwitch) ──────────
        bw_bytes_per_s = node.get("inter_node_bw", node["intra_node_bw"])
        latency_s = node.get("p2p_latency", 0.0)
        auto_topology = "Switch"
        tier = "intra-rack"

    elif num_gpus_per_rack is not None and num_gpus > num_gpus_per_rack:
        # ── Tier 3: inter-rack (InfiniBand) ──────────────────────
        bw_bytes_per_s = node.get("inter_rack_bw", node.get("inter_node_bw", node["intra_node_bw"]))
        latency_s = node.get("inter_rack_latency", node.get("p2p_latency", 0.0))
        auto_topology = "Ring"
        tier = "inter-rack"

    else:
        # ── 2-tier fallback: inter-node (IB or NVLink between nodes)
        bw_bytes_per_s = node.get("inter_node_bw", node["intra_node_bw"])
        latency_s = node.get("p2p_latency", 0.0)
        auto_topology = "Ring"
        tier = "inter-node"

    bandwidth_gbps = bw_bytes_per_s / 1e9  # Bytes/s → GB/s
    latency_ns = latency_s * 1e9            # seconds → ns
    # AstraSim requires at least some latency
    if latency_ns < 1.0:
        latency_ns = 500.0

    return {
        "npus_count": num_gpus,
        "bandwidth_gbps": bandwidth_gbps,
        "latency_ns": latency_ns,
        "topology": topology or auto_topology,
        "tier": tier,
    }


# ---------------------------------------------------------------------------
# AstraSimManager – unified simulation interface
# ---------------------------------------------------------------------------
class AstraSimManager:
    """Centralized ASTRA-Sim network simulation manager.

    Includes all ASTRA-Sim util. (topology creation, P2P,
    collectives, KV-cache transfer) behind a clean Python API.

    Construction parameters
    -----------------------
    * *system_spec* – parsed system YAML dict; used to auto-derive
      bandwidth / latency / topology from the hardware description.
    * *network_config* – explicit path to a static AstraSim YAML file.
      When given, this file is used directly (no auto-generation).
    * *topology_type* – default topology when auto-generating.
    * *cache_dir* – directory for auto-generated YAML files.

    If neither *system_spec* nor *network_config* is provided, the
    manager falls back to ``DEFAULT_NETWORK_CONFIG`` (Ring.yml).
    """

    def __init__(
        self,
        system_spec: dict | None = None,
        network_config: str | None = None,
        topology_type: str = "Ring",
        cache_dir: str | None = None,
    ) -> None:
        self._system_spec = system_spec
        self._explicit_config = network_config
        self._default_topology_type = topology_type
        self._cache_dir = cache_dir or _DEFAULT_TOPOLOGY_CACHE_DIR
        self._enabled = NETWORK_SIM_AVAILABLE

        if not self._enabled:
            logger.warning(
                "ASTRA-Sim Manager created but ASTRA-Sim library is not available. "
                "All simulation calls will return fallback values."
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """True when the ASTRA-Sim C++ library is loaded."""
        return self._enabled

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------
    def get_topology_config(self, num_gpus: int) -> str:
        """Return path to a topology YAML suitable for *num_gpus*.

        * If an explicit *network_config* was passed at construction time,
          it is returned directly (caller is responsible for ensuring the
          file has enough NPUs).
        * If a *system_spec* was provided, a topology YAML is
          auto-generated (or reused from cache) with correct bandwidth,
          latency, and NPU count.
        * Otherwise falls back to ``DEFAULT_NETWORK_CONFIG``.
        """
        if self._explicit_config is not None:
            return self._explicit_config

        if self._system_spec is not None:
            params = derive_network_params_from_system_spec(
                self._system_spec, num_gpus, topology=self._default_topology_type
            )
            return get_or_create_topology_config(
                npus_count=params["npus_count"],
                bandwidth_gbps=params["bandwidth_gbps"],
                latency_ns=params["latency_ns"],
                topology=params["topology"],
                cache_dir=self._cache_dir,
            )

        return DEFAULT_NETWORK_CONFIG

    def _build_topology(self, num_gpus: int):
        """Create a fresh (EventQueue, Topology) pair for *num_gpus*.

        Returns ``(event_queue, topology, config_path)`` or raises
        ``RuntimeError`` when AstraSim is unavailable.
        """
        if not self._enabled:
            raise RuntimeError("AstraSim is not available")

        config_path = self.get_topology_config(num_gpus)
        network_parser = NetworkParser(config_path)
        event_queue = EventQueue()
        Topology.set_event_queue(event_queue)
        topology = construct_topology(network_parser)
        return event_queue, topology, config_path

    # ------------------------------------------------------------------
    # Multi-tier topology helpers
    # ------------------------------------------------------------------
    def _classify_tier(
        self, src_gpu: int, dst_gpu: int
    ) -> tuple[tuple[str, int], int, int]:
        """Classify a GPU-to-GPU transfer into a network tier.

        Returns ``(tier_key, local_src, local_dst)`` where:

        * *tier_key* = ``(tier_name, group_id)`` uniquely identifies an
          independent congestion domain.  Distinct group IDs within the
          same tier have physically separate links, so they can't
          congest each other (e.g. NVLink inside node 0 vs node 1).
        * *local_src* / *local_dst* are NPU indices inside the
          topology created for that tier+group.

        Tier classification:

        * **intra-node** – both GPUs on the same node → NVLink topology.
          Group = node index.
        * **intra-rack** – same rack, different nodes (3-tier systems
          like GB200 with ``num_gpus_per_rack``) → NVSwitch topology.
          Group = rack index.
        * **inter-node** – different nodes, 2-tier system → IB fabric.
          Group = 0 (global).
        * **inter-rack** – different racks (3-tier) → IB fabric.
          Group = 0 (global).
        * **flat** – no ``system_spec`` → raw GPU IDs, single topology.
        """
        if self._system_spec is None:
            return ("flat", 0), src_gpu, dst_gpu

        node = self._system_spec["node"]
        gpn = node["num_gpus_per_node"]  # GPUs per node
        gpr = node.get("num_gpus_per_rack")  # GPUs per rack (None for 2-tier)

        src_node = src_gpu // gpn
        dst_node = dst_gpu // gpn

        if src_node == dst_node:
            # Intra-node: each node is an independent congestion domain
            return ("intra-node", src_node), src_gpu % gpn, dst_gpu % gpn

        if gpr is not None:
            npn = gpr // gpn  # nodes per rack
            src_rack = src_node // npn
            dst_rack = dst_node // npn

            if src_rack == dst_rack:
                # Intra-rack / inter-node via NVSwitch: per-rack domain
                return (
                    ("intra-rack", src_rack),
                    src_node % npn,
                    dst_node % npn,
                )
            else:
                # Inter-rack IB: global domain
                return ("inter-rack", 0), src_rack, dst_rack

        # 2-tier system: inter-node IB fabric (one global domain)
        return ("inter-node", 0), src_node, dst_node

    def _tier_topology_params(self, tier_name: str, num_units: int) -> dict:
        """Return AstraSim topology parameters for a given tier.

        Args:
            tier_name: ``"intra-node"``, ``"intra-rack"``,
                ``"inter-node"``, ``"inter-rack"``, or ``"flat"``.
            num_units: Number of NPUs the topology must contain.

        Returns:
            dict with ``npus_count``, ``bandwidth_gbps``,
            ``latency_ns``, ``topology`` keys.
        """
        if tier_name == "flat" or self._system_spec is None:
            return {
                "npus_count": num_units,
                "bandwidth_gbps": 25.0,
                "latency_ns": 500.0,
                "topology": "Ring",
            }

        node = self._system_spec["node"]

        if tier_name == "intra-node":
            bw = node["intra_node_bw"] / 1e9
            lat = node.get("p2p_latency", 0.0) * 1e9
            topo = "FullyConnected"
        elif tier_name == "intra-rack":
            bw = node.get("inter_node_bw", node["intra_node_bw"]) / 1e9
            lat = node.get("p2p_latency", 0.0) * 1e9
            topo = "Switch"
        elif tier_name == "inter-node":
            bw = node.get("inter_node_bw", node["intra_node_bw"]) / 1e9
            lat = node.get("p2p_latency", 0.0) * 1e9
            topo = "Switch"
        elif tier_name == "inter-rack":
            bw = node.get(
                "inter_rack_bw",
                node.get("inter_node_bw", node["intra_node_bw"]),
            ) / 1e9
            lat = node.get(
                "inter_rack_latency",
                node.get("p2p_latency", 0.0),
            ) * 1e9
            topo = "Switch"
        else:
            raise ValueError(f"Unknown tier: {tier_name}")

        if lat < 1.0:
            lat = 500.0

        return {
            "npus_count": num_units,
            "bandwidth_gbps": bw,
            "latency_ns": lat,
            "topology": topo,
        }

    def _simulate_tiered_transfers(
        self,
        transfers: list[tuple[int, int, int]],
    ) -> float:
        """Simulate a batch of GPU-to-GPU transfers using tiered topologies.

        Transfers are grouped by ``(tier_name, group_id)``.  Each group
        gets its own 1-D AstraSim topology and ``EventQueue`` so that
        concurrent transfers **within** the same tier+group contend for
        bandwidth (congestion), while transfers on **different** physical
        links (different tiers or groups) run independently.

        The returned latency is the **maximum** across all groups – the
        overall transfer completes when the slowest group finishes.

        Args:
            transfers: List of ``(src_gpu, dst_gpu, size_bytes)`` tuples.

        Returns:
            Simulated latency in **milliseconds**.

        Raises:
            RuntimeError: If AstraSim is not available.
        """
        if not self._enabled:
            raise RuntimeError("AstraSim is not available")

        if not transfers:
            return 0.0

        # ── 1. Group transfers by tier+group ────────────────────────
        tier_groups: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
        for src_gpu, dst_gpu, size_bytes in transfers:
            if src_gpu == dst_gpu:
                continue
            tier_key, local_src, local_dst = self._classify_tier(
                src_gpu, dst_gpu
            )
            tier_groups.setdefault(tier_key, []).append(
                (local_src, local_dst, size_bytes)
            )

        if not tier_groups:
            return 0.0

        # ── 2. Simulate each group independently ────────────────────
        tier_latencies: dict[tuple[str, int], float] = {}

        for tier_key, group in tier_groups.items():
            tier_name, group_id = tier_key

            # Determine topology size from max local ID
            max_local_id = max(max(s, d) for s, d, _ in group)
            num_units = max_local_id + 1

            params = self._tier_topology_params(tier_name, num_units)
            config_path = get_or_create_topology_config(
                npus_count=params["npus_count"],
                bandwidth_gbps=params["bandwidth_gbps"],
                latency_ns=params["latency_ns"],
                topology=params["topology"],
                cache_dir=self._cache_dir,
            )

            network_parser = NetworkParser(config_path)
            event_queue = EventQueue()
            Topology.set_event_queue(event_queue)
            topology = construct_topology(network_parser)

            topo_size = topology.get_npus_count()

            for i, (local_src, local_dst, size_bytes) in enumerate(group):
                if max(local_src, local_dst) >= topo_size:
                    logger.warning(
                        f"Tier {tier_name} group {group_id}: local IDs "
                        f"({local_src}, {local_dst}) exceed topology size "
                        f"{topo_size}. Skipping."
                    )
                    continue
                chunk = Chunk.create_with_event_queue(
                    size_bytes,
                    local_src,
                    local_dst,
                    i,
                    topology,
                    event_queue,
                )
                topology.send_python(chunk)

            while not event_queue.finished():
                event_queue.proceed()

            latency_ms = event_queue.get_current_time() / 1e6
            tier_latencies[tier_key] = latency_ms

            logger.debug(
                f"Tier {tier_name} (group {group_id}): "
                f"{len(group)} transfers → {latency_ms:.4f} ms"
            )

        # ── 3. Overall latency = slowest tier/group ─────────────────
        return max(tier_latencies.values()) if tier_latencies else 0.0

    def _build_worker_transfer_plan(
        self,
        src_pp_stages: list[list[int]],
        dst_pp_stages: list[list[int]],
        kv_cache_size: int,
        *,
        p_worker: int,
        d_worker: int,
    ) -> list[dict]:
        """Build a PP+TP-aware KV transfer plan for one prefill→decode worker pair.

        Accepts the **full** worker-level ``pp_stages`` and total
        ``kv_cache_size`` (across all layers, before any PP or TP
        sharding).  Both PP-stage splitting and TP-rank splitting are
        handled internally:

        1. **PP splitting** – The total KV is divided across PP stage
           pairs using ``lcm(src_pp, dst_pp)`` units.  When
           ``src_pp == dst_pp`` (common case) this is a simple 1:1
           mapping; stage *k* on the prefill side transfers to stage *k*
           on the decode side, each carrying ``kv_cache_size / pp``.

        2. **TP splitting** – Within each PP-stage pair the per-stage
           KV bytes are further divided across TP-rank pairs using
           ``lcm(src_tp, dst_tp)`` units.  When ``src_tp == dst_tp``
           each TP rank sends its shard to the matching decode rank;
           otherwise cross-rank resharding transfers are generated.

        The result is a flat list of concrete GPU-to-GPU transfer dicts,
        each with ``src``, ``dst``, ``bytes``, ``mode``, and worker/shard
        metadata.

        Args:
            src_pp_stages: Prefill worker's PP stages – a list of
                ``pp_size`` lists, each containing ``tp_size`` GPU IDs.
            dst_pp_stages: Decode worker's PP stages (same structure).
            kv_cache_size: Total logical KV cache bytes for this worker
                (full model, before PP/TP sharding).
            p_worker: Prefill worker index (metadata only).
            d_worker: Decode worker index (metadata only).

        Returns:
            List of transfer dicts with keys ``p_worker``, ``d_worker``,
            ``pp_stage``, ``shard``, ``src``, ``dst``, ``bytes``, ``mode``.
        """
        if kv_cache_size <= 0:
            return []

        src_pp = len(src_pp_stages)
        dst_pp = len(dst_pp_stages)
        if src_pp == 0 or dst_pp == 0:
            return []

        # ── Step 1: Partition KV across PP-stage pairs ───────────────
        pp_unit_count = math.lcm(src_pp, dst_pp)
        src_pp_units = pp_unit_count // src_pp   # units per src PP stage
        dst_pp_units = pp_unit_count // dst_pp   # units per dst PP stage

        kv_per_pp_unit_base = kv_cache_size // pp_unit_count
        kv_per_pp_unit_rem = kv_cache_size % pp_unit_count

        pp_pair_bytes: dict[tuple[int, int], int] = {}
        for pp_unit in range(pp_unit_count):
            unit_bytes = kv_per_pp_unit_base + (1 if pp_unit < kv_per_pp_unit_rem else 0)
            if unit_bytes == 0:
                continue
            src_pp_rank = pp_unit // src_pp_units
            dst_pp_rank = pp_unit // dst_pp_units
            key = (src_pp_rank, dst_pp_rank)
            pp_pair_bytes[key] = pp_pair_bytes.get(key, 0) + unit_bytes

        # ── Step 2: For each PP-stage pair, split across TP ranks ────
        transfers: list[dict] = []

        for (src_pp_rank, dst_pp_rank), pp_kv_bytes in sorted(pp_pair_bytes.items()):
            src_stage_gpus = src_pp_stages[src_pp_rank]
            dst_stage_gpus = dst_pp_stages[dst_pp_rank]
            src_tp = len(src_stage_gpus)
            dst_tp = len(dst_stage_gpus)

            if src_tp == 0 or dst_tp == 0:
                continue

            # TP-level overlap reshard via lcm partitioning
            tp_unit_count = math.lcm(src_tp, dst_tp)
            src_tp_units = tp_unit_count // src_tp
            dst_tp_units = tp_unit_count // dst_tp
            tp_unit_base = pp_kv_bytes // tp_unit_count
            tp_unit_rem = pp_kv_bytes % tp_unit_count

            tp_pair_bytes: dict[tuple[int, int], int] = {}
            for tp_unit in range(tp_unit_count):
                unit_bytes = tp_unit_base + (1 if tp_unit < tp_unit_rem else 0)
                if unit_bytes == 0:
                    continue
                src_idx = tp_unit // src_tp_units
                dst_idx = tp_unit // dst_tp_units
                tp_pair_bytes[(src_idx, dst_idx)] = (
                    tp_pair_bytes.get((src_idx, dst_idx), 0) + unit_bytes
                )

            mode = "paired" if src_tp == dst_tp else "tp_overlap_reshard"
            for shard_idx, ((src_idx, dst_idx), size_bytes) in enumerate(
                sorted(tp_pair_bytes.items())
            ):
                transfers.append({
                    "p_worker": p_worker,
                    "d_worker": d_worker,
                    "pp_stage": (src_pp_rank, dst_pp_rank),
                    "shard": shard_idx,
                    "src": src_stage_gpus[src_idx],
                    "dst": dst_stage_gpus[dst_idx],
                    "bytes": size_bytes,
                    "mode": mode,
                })

        return transfers

    def build_kv_transfer_plan(
        self,
        gpu_layout: dict,
        kv_cache_size: int,
        prefill_batch_size: int,
    ) -> list[dict]:
        """Build the detailed prefill→decode KV transfer plan.

        The pairing between prefill and decode workers is read from
        ``gpu_layout["prefill_decode_pairing"]`` — a ``dict[int, list[int]]``
        mapping each prefill worker index to a list of decode worker
        indices it sends KV to.

        * **P >= D** — each prefill maps to exactly one decode worker
          (single-element list).  Multiple prefills may share the same
          decode target.
        * **P < D** — each prefill fans out to ``ceil(D/P)`` decode
          workers.  The prefill's ``kv_cache_size`` is divided evenly
          among its targets (each decode gets a subset of sequences).

        PP-stage-level and TP-rank-level splitting are handled by
        ``_build_worker_transfer_plan``: the full ``pp_stages`` and
        the per-target ``kv_cache_size`` are passed in, and the method
        partitions bytes across all PP stage pairs and TP rank pairs.

        Returns one dict per concrete GPU-to-GPU transfer with
        source/destination GPU IDs, byte count, worker assignment, and
        transfer mode.
        """
        prefill_worker_layouts = gpu_layout["prefill_worker_layouts"]
        decode_worker_layouts = gpu_layout["decode_worker_layouts"]
        num_prefill_workers = len(prefill_worker_layouts)
        num_decode_workers = len(decode_worker_layouts)

        if num_prefill_workers == 0 or num_decode_workers == 0:
            return []

        pairing = gpu_layout["prefill_decode_pairing"]

        transfers = []

        for p_idx in range(num_prefill_workers):
            d_indices = pairing[p_idx]  # list[int]
            num_targets = len(d_indices)

            # Divide KV evenly across decode targets for this prefill.
            kv_per_target_base = kv_cache_size // num_targets
            kv_per_target_rem = kv_cache_size % num_targets

            for target_idx, d_idx in enumerate(d_indices):
                target_kv = kv_per_target_base + (
                    1 if target_idx < kv_per_target_rem else 0
                )
                prefill_worker = prefill_worker_layouts[p_idx]
                decode_worker = decode_worker_layouts[d_idx]

                # ── AFD DP-replica-aware path ────────────────────────
                #
                # When AFD is enabled, only attention GPUs participate
                # in KV cache transfer.  Each DP replica holds KV for
                # ``batch_size / dp_size`` sequences, so KV bytes are
                # divided by dp_size.  We iterate over DP replicas
                # independently, using ``attn_dp_pp_stages[dp_rank]``
                # as the PP stages for each replica.
                p_afd = prefill_worker.get("enable_afd", False)
                d_afd = decode_worker.get("enable_afd", False)
                p_dp = prefill_worker.get("attn_dp_size", 1)
                d_dp = decode_worker.get("attn_dp_size", 1)

                if p_afd and d_afd and p_dp > 1 and d_dp > 1:
                    # Both workers are AFD-enabled with DP replicas.
                    # DP sizes must match for a valid pairing.
                    dp_size = min(p_dp, d_dp)
                    kv_per_dp_base = target_kv // dp_size
                    kv_per_dp_rem = target_kv % dp_size

                    p_dp_stages = prefill_worker["attn_dp_pp_stages"]
                    d_dp_stages = decode_worker["attn_dp_pp_stages"]

                    for dp_rank in range(dp_size):
                        kv_per_dp_replica = kv_per_dp_base + (
                            1 if dp_rank < kv_per_dp_rem else 0
                        )
                        pair_transfers = self._build_worker_transfer_plan(
                            p_dp_stages[dp_rank],
                            d_dp_stages[dp_rank],
                            kv_per_dp_replica,
                            p_worker=p_idx,
                            d_worker=d_idx,
                        )
                        # Tag each transfer with the DP rank for
                        # traceability.
                        for t in pair_transfers:
                            t["dp_rank"] = dp_rank
                        transfers.extend(pair_transfers)
                else:
                    # Non-AFD or dp=1 — use pp_stages directly.
                    # For AFD with dp=1, pp_stages is already set to
                    # attn_dp_pp_stages[0] so this works correctly.
                    pair_transfers = self._build_worker_transfer_plan(
                        prefill_worker["pp_stages"],
                        decode_worker["pp_stages"],
                        target_kv,
                        p_worker=p_idx,
                        d_worker=d_idx,
                    )
                    transfers.extend(pair_transfers)

        return transfers

    # ------------------------------------------------------------------
    # Simulation APIs
    # ------------------------------------------------------------------
    def simulate_p2p(
        self,
        message_size_bytes: int,
        src_gpu: int = 0,
        dst_gpu: int = 1,
        num_total_gpus: int | None = None,
    ) -> float | None:
        """Simulate a point-to-point transfer.

        When a ``system_spec`` is available the transfer is automatically
        routed through the correct network tier (intra-node NVLink,
        inter-node IB, etc.) based on the source/destination GPU IDs.

        Args:
            message_size_bytes: Payload in bytes.
            src_gpu: Source GPU ID.
            dst_gpu: Destination GPU ID.
            num_total_gpus: Total GPUs in the topology.  When ``None``,
                defaults to ``max(src_gpu, dst_gpu) + 1``.

        Returns:
            Simulated latency in **milliseconds**, or ``None`` on failure.
        """
        if not self._enabled:
            return None

        if num_total_gpus is None:
            num_total_gpus = max(src_gpu, dst_gpu) + 1

        try:
            if self._system_spec is not None:
                # ── Multi-tier: pick the correct tier for this pair ──
                latency_ms = self._simulate_tiered_transfers(
                    [(src_gpu, dst_gpu, message_size_bytes)]
                )
                logger.debug(
                    f"AstraSim P2P (tiered): {message_size_bytes} bytes, "
                    f"GPU {src_gpu} → GPU {dst_gpu} → {latency_ms:.4f} ms"
                )
                return latency_ms

            # ── Legacy single-topology path ──────────────────────────
            event_queue, topology, _ = self._build_topology(num_total_gpus)

            # Validate GPU IDs fit in topology
            topo_size = topology.get_npus_count()
            if max(src_gpu, dst_gpu) >= topo_size:
                logger.warning(
                    f"P2P GPU IDs ({src_gpu}, {dst_gpu}) exceed topology size "
                    f"({topo_size}). Returning None."
                )
                return None

            chunk = Chunk.create_with_event_queue(
                message_size_bytes, src_gpu, dst_gpu, 0, topology, event_queue,
            )
            topology.send_python(chunk)

            while not event_queue.finished():
                event_queue.proceed()

            latency_ms = event_queue.get_current_time() / 1e6
            logger.debug(
                f"AstraSim P2P: {message_size_bytes} bytes, "
                f"GPU {src_gpu} → GPU {dst_gpu} → {latency_ms:.4f} ms"
            )
            return latency_ms

        except Exception as e:
            logger.warning(f"AstraSim P2P simulation failed: {e}")
            return None

    def simulate_collective(
        self,
        message_size_bytes: int,
        num_gpus: int,
        operation: str,
    ) -> float | None:
        """Simulate a collective communication operation.

        Args:
            message_size_bytes: Total message payload in **bytes**.
            num_gpus: Number of GPUs participating.
            operation: One of ``"all_reduce"``, ``"all_gather"``,
                ``"reduce_scatter"``, ``"alltoall"``.

        Returns:
            Simulated latency in **milliseconds**, or ``None`` on failure.
        """
        if not self._enabled:
            return None

        try:
            event_queue, topology, _ = self._build_topology(num_gpus)
            npus_count = min(topology.get_npus_count(), num_gpus)
            devices_count = min(topology.get_devices_count(), num_gpus)

            request_id = 0

            if operation in ("all_gather", "reduce_scatter", "alltoall"):
                per_gpu_bytes = message_size_bytes // max(npus_count, 1)
                for src in range(npus_count):
                    for dst in range(devices_count):
                        if src == dst:
                            continue
                        chunk = Chunk.create_with_event_queue(
                            per_gpu_bytes, src, dst, request_id, topology, event_queue,
                        )
                        topology.send_python(chunk)
                        request_id += 1
            elif operation == "all_reduce":
                per_gpu_bytes = message_size_bytes // max(npus_count, 1)
                for src in range(npus_count):
                    for dst in range(devices_count):
                        if src == dst:
                            continue
                        chunk = Chunk.create_with_event_queue(
                            per_gpu_bytes * 2, src, dst, request_id, topology, event_queue,
                        )
                        topology.send_python(chunk)
                        request_id += 1
            else:
                raise ValueError(f"Unsupported collective operation: {operation}")

            while not event_queue.finished():
                event_queue.proceed()

            latency_ms = event_queue.get_current_time() / 1e6
            logger.debug(
                f"AstraSim collective {operation}: {message_size_bytes} bytes, "
                f"{num_gpus} GPUs → {latency_ms:.4f} ms"
            )
            return latency_ms

        except Exception as e:
            logger.warning(
                f"AstraSim collective simulation failed for {operation}: {e}"
            )
            return None

    def simulate_kv_cache_transfer(
        self,
        gpu_layout: dict,
        kv_cache_size: int,
        prefill_batch_size: int,
    ) -> float:
        """Simulate KV-cache transfer latency for disaggregated inference.

        When a ``system_spec`` is available, transfers are automatically
        classified into network tiers based on the GPU layout.  For
        example, prefill GPU 7 → decode GPU 8 on an 8-GPU-per-node
        system routes through the **inter-node** tier (IB bandwidth),
        while GPU 2 → GPU 5 on the same node uses the **intra-node**
        tier (NVLink bandwidth).  Transfers within the same tier share
        an ``EventQueue`` and therefore model congestion.

        Args:
            gpu_layout: GPU assignment dict from
                ``DisaggInferenceSession._build_gpu_layout``.
            kv_cache_size: Total KV cache size in bytes.
            prefill_batch_size: Number of sequences in the batch.

        Returns:
            Network transfer latency in **milliseconds** (0.0 on failure
            or when AstraSim is unavailable).
        """
        if not self._enabled:
            return 0.0

        prefill_workers = gpu_layout["prefill_workers"]
        decode_workers = gpu_layout["decode_workers"]
        transfer_plan = self.build_kv_transfer_plan(
            gpu_layout=gpu_layout,
            kv_cache_size=kv_cache_size,
            prefill_batch_size=prefill_batch_size,
        )
        transfers: list[tuple[int, int, int]] = [
            (t["src"], t["dst"], t["bytes"]) for t in transfer_plan if t["bytes"] > 0
        ]

        if not transfers:
            return 0.0

        try:
            if self._system_spec is not None:
                # ── Multi-tier simulation ────────────────────────────
                latency_ms = self._simulate_tiered_transfers(transfers)
                logger.info(
                    f"AstraSim KV cache transfer (tiered): "
                    f"{latency_ms:.3f} ms for {len(transfers)} transfers, total {kv_cache_size} bytes"
                )
                return latency_ms

            # ── Legacy single-topology path ──────────────────────────
            max_gpu_id = max(
                max(g for w in prefill_workers for g in w),
                max(g for w in decode_workers for g in w),
            )
            num_total_gpus = max_gpu_id + 1

            event_queue, topology, config_path = self._build_topology(
                num_total_gpus
            )

            topo_size = topology.get_npus_count()
            if max_gpu_id >= topo_size:
                logger.warning(
                    f"GPU layout requires {num_total_gpus} nodes but topology "
                    f"only has {topo_size} (config: {config_path}). "
                    f"Skipping network simulation."
                )
                return 0.0

            chunk_id = 0
            for src_gpu, dst_gpu, size_bytes in transfers:
                chunk = Chunk.create_with_event_queue(
                    size_bytes, src_gpu, dst_gpu, chunk_id,
                    topology, event_queue,
                )
                topology.send_python(chunk)
                logger.debug(
                    f"KV transfer chunk {chunk_id}: GPU {src_gpu} -> "
                    f"GPU {dst_gpu}, {size_bytes} bytes"
                )
                chunk_id += 1

            while not event_queue.finished():
                event_queue.proceed()

            latency_ms = event_queue.get_current_time() / 1e6
            logger.info(
                f"AstraSim KV cache transfer: {latency_ms:.3f} ms "
                f"for {chunk_id} chunks, total {kv_cache_size} bytes"
            )
            return latency_ms

        except Exception as e:
            logger.warning(f"AstraSim KV cache transfer simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
