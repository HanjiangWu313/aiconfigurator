"""
AstraSim network simulator utilities.

Provides :class:`AstraSimManager` – a single entry-point for every
network-simulation need (P2P, collectives, KV-cache transfer).

Topology YAML files are generated **on-the-fly** from system-spec
parameters and cached on disk so duplicate files are never created.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants to the astrasim library
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

# Congestion-unaware multi-dim topology support
NETWORK_SIM_UNAWARE_AVAILABLE = False
_UnAwareNetworkParser = None
_unaware_construct_topology = None

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

try:
    from simulation_py_congestion_unaware import (
        NetworkParser as _UnAwareNetworkParser,
        construct_topology as _unaware_construct_topology,
    )
    NETWORK_SIM_UNAWARE_AVAILABLE = True
    logger.info(
        f"AstraSim congestion-unaware simulator loaded from {_NETWORK_SIM_LIB_PATH}"
    )
except ImportError as e:
    NETWORK_SIM_UNAWARE_AVAILABLE = False
    logger.debug(f"AstraSim congestion-unaware simulator not available: {e}")


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
    return f"auto_{topology}_{npus_count}npus_{bandwidth_gbps}gbps_{latency_ns}ns.yml"


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

    Targets NVIDIA GPU systems with up to three network tiers 
    (TODO: Update for other vendors accordingly):

    * **Tier 1 – intra-node** (``intra_node_bw``, NVLink):
      ``num_gpus <= num_gpus_per_node`` → ``Switch``
    * **Tier 2 – intra-rack** (``inter_node_bw``, NVSwitch, for 3-tier systems only):
      ``num_gpus <= num_gpus_per_rack`` (3-tier systems only) → ``Switch``
    * **Tier 3 – inter-rack / inter-node** (``inter_rack_bw`` / ``inter_node_bw``, IB):
      all remaining cases → ``Switch``

    Tier examples:

    * **H100 / B200 SXM** (NVL8, 2-tier):
      ≤ 8 GPUs → NVLink (intra-node) · > 8 GPUs → IB (inter-node)
    * **GB200 SXM** (NVL4 + NVL72, 3-tier):
      ≤ 4 GPUs → NVLink (intra-node) · ≤ 72 GPUs → NVSwitch (intra-rack) · > 72 GPUs → IB (inter-rack)

    Args:
        system_spec: Parsed system YAML dict (must contain ``node`` key).
        num_gpus: Total number of GPUs in the topology.
        topology: Override topology type; ``None`` → auto-select per tier.

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
        auto_topology = "Switch"
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
        auto_topology = "Switch"
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


def derive_multidim_network_params(
    system_spec: dict,
    num_gpus: int,
) -> dict:
    """Derive a **multi-dimensional** AstraSim topology from *system_spec*.

    Instead of picking a single tier, this builds a topology with one
    dimension per network tier that is actually used.

    Examples (contiguous GPU allocation):

    * **H100 SXM, 8 GPUs** (1 node) → 1-dim ``Switch(8, 450 GB/s)``
    * **H100 SXM, 16 GPUs** (2 nodes) → 2-dim:
      dim 0 = ``Switch(8, 450 GB/s)``   (intra-node, NVLink)
      dim 1 = ``Switch(2, 25 GB/s)``    (inter-node, IB)
    * **GB200 NVL72, 144 GPUs** (2 racks) → 3-dim:
      dim 0 = ``Switch(4, 900 GB/s)``   (intra-node, NVLink)
      dim 1 = ``Switch(18, 200 GB/s)``  (intra-rack, NVSwitch)
      dim 2 = ``Switch(2, 50 GB/s)``    (inter-rack, IB)

    Returns:
        dict with list-valued keys: ``topologies``, ``npus_counts``,
        ``bandwidths_gbps``, ``latencies_ns`` (one entry per dimension),
        plus ``total_npus`` (product of npus_counts).
    """
    node = system_spec["node"]
    gpn = node["num_gpus_per_node"]
    gpr = node.get("num_gpus_per_rack")  # None for 2-tier systems

    def _bw_gbps(key: str, fallback_key: str | None = None) -> float:
        bw = node.get(key, node.get(fallback_key, node["intra_node_bw"]) if fallback_key else node["intra_node_bw"])
        return bw / 1e9

    def _lat_ns(key: str = "p2p_latency", fallback_key: str | None = None) -> float:
        lat = node.get(key, node.get(fallback_key, 0.0) if fallback_key else 0.0)
        lat_ns = lat * 1e9
        return max(lat_ns, 500.0)  # AstraSim needs at least some latency

    topologies: list[str] = []
    npus_counts: list[int] = []
    bandwidths_gbps: list[float] = []
    latencies_ns: list[float] = []

    # ── Dim 0: intra-node (NVLink) ─────────────────────────────
    intra_node_units = min(gpn, num_gpus)
    topologies.append("Switch")
    npus_counts.append(intra_node_units)
    bandwidths_gbps.append(_bw_gbps("intra_node_bw"))
    latencies_ns.append(_lat_ns())

    if num_gpus <= gpn:
        # All GPUs fit on one node — single dimension is enough.
        return {
            "topologies": topologies,
            "npus_counts": npus_counts,
            "bandwidths_gbps": bandwidths_gbps,
            "latencies_ns": latencies_ns,
            "total_npus": num_gpus,
        }

    if gpr is not None:
        # 3-tier system (e.g. GB200 NVL72)
        npn = gpr // gpn  # nodes per rack
        num_nodes = (num_gpus + gpn - 1) // gpn

        if num_nodes <= npn:
            # All nodes in one rack → 2 dims (intra-node + intra-rack)
            topologies.append("Switch")
            npus_counts.append(num_nodes)
            bandwidths_gbps.append(_bw_gbps("inter_node_bw", "intra_node_bw"))
            latencies_ns.append(_lat_ns())
        else:
            # Multiple racks → 3 dims
            topologies.append("Switch")
            npus_counts.append(npn)
            bandwidths_gbps.append(_bw_gbps("inter_node_bw", "intra_node_bw"))
            latencies_ns.append(_lat_ns())

            num_racks = (num_nodes + npn - 1) // npn
            topologies.append("Switch")
            npus_counts.append(num_racks)
            bandwidths_gbps.append(
                _bw_gbps("inter_rack_bw", "inter_node_bw")
            )
            latencies_ns.append(
                _lat_ns("inter_rack_latency", "p2p_latency")
            )
    else:
        # 2-tier system (e.g. H100 SXM NVL8)
        num_nodes = (num_gpus + gpn - 1) // gpn
        topologies.append("Switch")
        npus_counts.append(num_nodes)
        bandwidths_gbps.append(_bw_gbps("inter_node_bw", "intra_node_bw"))
        latencies_ns.append(_lat_ns())

    total_npus = 1
    for c in npus_counts:
        total_npus *= c

    return {
        "topologies": topologies,
        "npus_counts": npus_counts,
        "bandwidths_gbps": bandwidths_gbps,
        "latencies_ns": latencies_ns,
        "total_npus": total_npus,
    }


def get_or_create_multidim_topology_config(
    topologies: list[str],
    npus_counts: list[int],
    bandwidths_gbps: list[float],
    latencies_ns: list[float],
    cache_dir: str | None = None,
) -> str:
    """Return path to a **multi-dimensional** topology YAML, creating it if needed.

    Args:
        topologies: Topology type per dimension (e.g. ``["Switch", "Switch"]``).
        npus_counts: NPUs per dimension (e.g. ``[8, 2]``).
        bandwidths_gbps: Bandwidth per dimension in GB/s.
        latencies_ns: Latency per dimension in ns.
        cache_dir: Directory for cached files.

    Returns:
        Absolute path to the YAML file.
    """
    for t in topologies:
        if t not in _VALID_TOPOLOGIES:
            raise ValueError(f"Invalid topology '{t}'. Must be one of {_VALID_TOPOLOGIES}")

    cache_dir = cache_dir or _DEFAULT_TOPOLOGY_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Deterministic filename from all dimensions
    parts = []
    for t, n, b, l in zip(topologies, npus_counts, bandwidths_gbps, latencies_ns):
        parts.append(f"{t}{n}_{b}gbps_{l}ns")
    filename = f"auto_multidim_{'_'.join(parts)}.yml"
    filepath = os.path.join(cache_dir, filename)

    if os.path.isfile(filepath):
        logger.debug(f"Reusing cached multi-dim topology config: {filepath}")
        return filepath

    topo_str = ", ".join(topologies)
    npu_str = ", ".join(str(n) for n in npus_counts)
    bw_str = ", ".join(str(b) for b in bandwidths_gbps)
    lat_str = ", ".join(str(l) for l in latencies_ns)

    content = (
        f"# Auto-generated multi-dim AstraSim topology config\n"
        f"topology: [ {topo_str} ]\n"
        f"npus_count: [ {npu_str} ]\n"
        f"bandwidth: [ {bw_str} ]  # GB/s\n"
        f"latency: [ {lat_str} ]  # ns\n"
    )

    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".yml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(f"Created multi-dim topology config: {filepath}")
    return filepath


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
    # Congestion-unaware multi-dim topology
    # ------------------------------------------------------------------
    @property
    def multidim_enabled(self) -> bool:
        """True when the congestion-unaware multi-dim simulator is loaded."""
        return NETWORK_SIM_UNAWARE_AVAILABLE

    def _build_multidim_topology(self, num_gpus: int):
        """Create a congestion-unaware multi-dim topology for *num_gpus*.

        When a ``system_spec`` is available, generates a multi-dim YAML
        with one dimension per network tier.  Otherwise falls back to a
        single-dim topology.

        Returns ``(topology, config_path)`` or raises ``RuntimeError``.
        """
        if not NETWORK_SIM_UNAWARE_AVAILABLE:
            raise RuntimeError("Congestion-unaware AstraSim is not available")

        if self._system_spec is not None:
            params = derive_multidim_network_params(self._system_spec, num_gpus)
            config_path = get_or_create_multidim_topology_config(
                topologies=params["topologies"],
                npus_counts=params["npus_counts"],
                bandwidths_gbps=params["bandwidths_gbps"],
                latencies_ns=params["latencies_ns"],
                cache_dir=self._cache_dir,
            )
        else:
            config_path = self.get_topology_config(num_gpus)

        parser = _UnAwareNetworkParser(config_path)
        topology = _unaware_construct_topology(parser)
        return topology, config_path

    def simulate_multidim_p2p(
        self,
        src_gpu: int,
        dst_gpu: int,
        message_size_bytes: int,
        num_gpus: int,
    ) -> float | None:
        """Simulate a single P2P transfer using the multi-dim topology.

        Calls ``topology.send(src, dst, bytes)`` which returns the
        latency for the **first** differing dimension only.  For
        transfers that span multiple dimensions, the caller must
        decompose the route explicitly.

        Args:
            src_gpu: Source GPU flat ID.
            dst_gpu: Destination GPU flat ID.
            message_size_bytes: Payload in bytes.
            num_gpus: Total GPUs to size the topology.

        Returns:
            Latency in **milliseconds**, or ``None`` on failure.
        """
        if not NETWORK_SIM_UNAWARE_AVAILABLE:
            return None

        try:
            topology, _ = self._build_multidim_topology(num_gpus)
            npus = topology.get_npus_count()
            if max(src_gpu, dst_gpu) >= npus:
                logger.warning(
                    f"GPU IDs ({src_gpu}, {dst_gpu}) exceed topology size {npus}"
                )
                return None
            delay_ns = topology.send(src_gpu, dst_gpu, message_size_bytes)
            return delay_ns / 1e6  # ns → ms
        except Exception as e:
            logger.warning(f"Multi-dim P2P simulation failed: {e}")
            return None

    def simulate_multidim_collective(
        self,
        message_size_bytes: int,
        num_gpus: int,
        operation: str,
        gpu_ids: list[int] | None = None,
    ) -> float | None:
        """Simulate a collective using hierarchical ring algorithms.

        Models collectives as **sequential phases per dimension**,
        mirroring how NCCL implements hierarchical ring all-reduce:

        For an ``all_reduce`` across a multi-dim topology the phases
        are:

        1. **Reduce-scatter** in the outermost (slowest) dimension.
        2. **Reduce-scatter** in the next dimension inward, and so on
           down to the innermost (fastest) dimension.
        3. **All-gather** back up from the innermost to the outermost
           dimension.

        Each phase uses a ring algorithm within its dimension:
        ``(N_dim − 1)`` sequential steps of ``chunk_bytes`` each,
        where ``chunk = remaining_data / N_dim``.  The latency of
        each step is obtained from the congestion-unaware
        ``topology.send()`` which returns
        ``hops × link_latency + chunk / bandwidth``.

        For ``all_gather`` / ``reduce_scatter`` the algorithm is the
        same but only the scatter or gather half is performed.

        For ``alltoall`` the algorithm uses ``N − 1`` sequential
        shift-exchange steps.

        Args:
            message_size_bytes: Total message payload in bytes.
            num_gpus: Number of participating GPUs.
            operation: ``"all_reduce"``, ``"all_gather"``,
                ``"reduce_scatter"``, or ``"alltoall"``.
            gpu_ids: Explicit GPU IDs; defaults to ``range(num_gpus)``.

        Returns:
            Latency in **milliseconds**, or ``None`` on failure.
        """
        if not NETWORK_SIM_UNAWARE_AVAILABLE:
            return None

        try:
            ids = gpu_ids if gpu_ids is not None else list(range(num_gpus))
            n = len(ids)
            if n < 2:
                return 0.0

            topology, _ = self._build_multidim_topology(num_gpus)
            npus_per_dim = topology.get_npus_count_per_dim()
            dims = topology.get_dims_count()

            total_ns = self._hierarchical_ring(
                topology, npus_per_dim, dims,
                message_size_bytes, operation,
            )

            latency_ms = total_ns / 1e6
            logger.debug(
                f"Multi-dim collective {operation}: "
                f"{message_size_bytes} bytes, {n} GPUs, "
                f"{dims} dims {npus_per_dim} → {latency_ms:.4f} ms"
            )
            return latency_ms

        except Exception as e:
            logger.warning(f"Multi-dim collective simulation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Hierarchical ring collective helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ring_latency_for_dim(
        topology,
        dim: int,
        npus_per_dim: list[int],
        chunk_bytes: int,
    ) -> int:
        """Compute the latency of one ring step in a given dimension.

        Picks a representative adjacent pair in *dim* (local IDs 0→1)
        with all other dimensions at coordinate 0 and calls
        ``topology.send()`` to get the per-hop cost at the correct
        bandwidth and latency for that dimension.

        Args:
            topology: Congestion-unaware ``Topology`` object.
            dim: Which dimension to evaluate.
            npus_per_dim: NPU counts per dimension.
            chunk_bytes: Bytes transferred per ring step.

        Returns:
            Latency in **nanoseconds** for one ring step.
        """
        # Build flat GPU IDs for the pair (0, …, 0) and (0, …, 1, …, 0)
        # where position *dim* differs.
        stride = 1
        for d in range(dim):
            stride *= npus_per_dim[d]
        src_flat = 0
        dst_flat = stride  # coord[dim] = 1, all others = 0
        return topology.send(src_flat, dst_flat, chunk_bytes)

    def _hierarchical_ring(
        self,
        topology,
        npus_per_dim: list[int],
        dims: int,
        message_size_bytes: int,
        operation: str,
    ) -> int:
        """Model a hierarchical ring collective across multiple dimensions.

        The algorithm proceeds dimension-by-dimension from the **outermost**
        (slowest, highest index) to the **innermost** (fastest, index 0).

        **all_reduce** = reduce-scatter all dims (outer→inner)
                       + all-gather all dims (inner→outer)

        **reduce_scatter** = reduce-scatter all dims (outer→inner)

        **all_gather** = all-gather all dims (inner→outer)

        **alltoall** = shift-exchange all dims (outer→inner)

        Within each dimension the ring algorithm takes ``(N_dim − 1)``
        sequential steps.  After reduce-scatter in a dimension the data
        per GPU shrinks by ``N_dim`` (each GPU holds 1/N_dim of the
        result), so the chunk size cascades correctly across dimensions.

        Returns:
            Total latency in **nanoseconds**.
        """
        total_ns = 0

        if operation == "alltoall":
            # all-to-all: each dim does (N_dim − 1) shift-exchange steps
            # with per-GPU payload = message / total_gpus
            total_gpus = 1
            for nd in npus_per_dim:
                total_gpus *= nd
            per_gpu = message_size_bytes // max(total_gpus, 1)
            for dim in reversed(range(dims)):
                n_dim = npus_per_dim[dim]
                if n_dim < 2:
                    continue
                steps = n_dim - 1
                step_ns = self._ring_latency_for_dim(
                    topology, dim, npus_per_dim, per_gpu,
                )
                total_ns += steps * step_ns
            return total_ns

        # ── Reduce-scatter phase (outer → inner) ─────────────────
        if operation in ("all_reduce", "reduce_scatter"):
            chunk = message_size_bytes
            for dim in reversed(range(dims)):
                n_dim = npus_per_dim[dim]
                if n_dim < 2:
                    continue
                chunk_per_step = chunk // n_dim
                steps = n_dim - 1
                step_ns = self._ring_latency_for_dim(
                    topology, dim, npus_per_dim, chunk_per_step,
                )
                total_ns += steps * step_ns
                chunk = chunk_per_step  # data shrinks

        # ── All-gather phase (inner → outer) ─────────────────────
        if operation in ("all_reduce", "all_gather"):
            # For all_gather-only, start from per-GPU share
            if operation == "all_gather":
                total_gpus = 1
                for nd in npus_per_dim:
                    total_gpus *= nd
                chunk = message_size_bytes // max(total_gpus, 1)
            # else: chunk is already set from reduce-scatter tail

            for dim in range(dims):
                n_dim = npus_per_dim[dim]
                if n_dim < 2:
                    continue
                # chunk is the current per-GPU data; send it per step
                steps = n_dim - 1
                step_ns = self._ring_latency_for_dim(
                    topology, dim, npus_per_dim, chunk,
                )
                total_ns += steps * step_ns
                chunk = chunk * n_dim  # data grows

        return total_ns

    def simulate_multidim_afd(
        self,
        attn_gpu_ids: list[int],
        ffn_gpu_ids: list[int],
        sender_bytes_per_gpu: int,
        receiver_bytes_per_gpu: int,
        pre_dispatch: bool = True,
        num_gpus: int | None = None,
    ) -> float | None:
        """Simulate AFD transfers using the congestion-unaware multi-dim topology.

        Each attn↔FFN P2P transfer is evaluated via ``send()`` with
        proper multi-tier bandwidth selection.  Because this topology API
        returns an independent point-to-point latency and does not model
        congestion, AFD fanout is modeled as endpoint serialization:
        transfers attached to the same sender or receiver add up, while
        independent endpoints progress concurrently.  The result is the
        slowest endpoint's accumulated latency.

        Args:
            attn_gpu_ids: Attention GPU IDs.
            ffn_gpu_ids: FFN GPU IDs.
            sender_bytes_per_gpu: Bytes each sender pushes.
            receiver_bytes_per_gpu: Bytes each receiver pulls.
            pre_dispatch: True for attn→FFN, False for FFN→attn.
            num_gpus: Total GPUs to size the topology.  If ``None``,
                uses ``max(all_ids) + 1``.

        Returns:
            Latency in **milliseconds**, or ``None`` on failure.
        """
        if not NETWORK_SIM_UNAWARE_AVAILABLE:
            return None
        if not attn_gpu_ids or not ffn_gpu_ids:
            return 0.0

        try:
            all_ids = list(attn_gpu_ids) + list(ffn_gpu_ids)
            total_gpus = num_gpus or (max(all_ids) + 1)
            topology, _ = self._build_multidim_topology(total_gpus)

            sender_elapsed_ns: dict[int, int] = {}
            receiver_elapsed_ns: dict[int, int] = {}
            max_pair_delay_ns = 0
            total_pair_delay_ns = 0

            def record_transfer(src_gpu: int, dst_gpu: int, size_bytes: int) -> None:
                nonlocal max_pair_delay_ns, total_pair_delay_ns
                delay = topology.send(src_gpu, dst_gpu, size_bytes)
                sender_elapsed_ns[src_gpu] = sender_elapsed_ns.get(src_gpu, 0) + delay
                receiver_elapsed_ns[dst_gpu] = receiver_elapsed_ns.get(dst_gpu, 0) + delay
                max_pair_delay_ns = max(max_pair_delay_ns, delay)
                total_pair_delay_ns += delay

            if pre_dispatch:
                N = len(ffn_gpu_ids)
                bytes_per_link = (
                    max(1, sender_bytes_per_gpu // N)
                    if sender_bytes_per_gpu > 0
                    else 0
                )
                self._log_afd_payload_debug(
                    direction="attn->FFN",
                    num_senders=len(attn_gpu_ids),
                    num_receivers=N,
                    sender_bytes_per_gpu=sender_bytes_per_gpu,
                    receiver_bytes_per_gpu=receiver_bytes_per_gpu,
                    bytes_per_link=bytes_per_link,
                    transfer_count=len(attn_gpu_ids) * N,
                )
                if bytes_per_link > 0:
                    for attn_gpu in attn_gpu_ids:
                        for ffn_gpu in ffn_gpu_ids:
                            record_transfer(attn_gpu, ffn_gpu, bytes_per_link)
            else:
                M = len(attn_gpu_ids)
                bytes_per_link = (
                    max(1, sender_bytes_per_gpu // M)
                    if sender_bytes_per_gpu > 0
                    else 0
                )
                self._log_afd_payload_debug(
                    direction="FFN->attn",
                    num_senders=len(ffn_gpu_ids),
                    num_receivers=M,
                    sender_bytes_per_gpu=sender_bytes_per_gpu,
                    receiver_bytes_per_gpu=receiver_bytes_per_gpu,
                    bytes_per_link=bytes_per_link,
                    transfer_count=len(ffn_gpu_ids) * M,
                )
                if bytes_per_link > 0:
                    for ffn_gpu in ffn_gpu_ids:
                        for attn_gpu in attn_gpu_ids:
                            record_transfer(ffn_gpu, attn_gpu, bytes_per_link)

            max_delay_ns = max(
                max(sender_elapsed_ns.values(), default=0),
                max(receiver_elapsed_ns.values(), default=0),
            )
            latency_ms = max_delay_ns / 1e6
            direction = "attn→FFN" if pre_dispatch else "FFN→attn"
            logger.debug(
                f"Multi-dim AFD ({direction}): "
                f"{len(attn_gpu_ids)} attn, {len(ffn_gpu_ids)} FFN → "
                f"{latency_ms:.4f} ms "
                f"(max_pair={max_pair_delay_ns / 1e6:.4f} ms, "
                f"sum_pairs={total_pair_delay_ns / 1e6:.4f} ms)"
            )
            return latency_ms

        except Exception as e:
            logger.warning(f"Multi-dim AFD simulation failed: {e}")
            return None

    def _log_afd_payload_debug(
        self,
        *,
        direction: str,
        num_senders: int,
        num_receivers: int,
        sender_bytes_per_gpu: int,
        receiver_bytes_per_gpu: int,
        bytes_per_link: int,
        transfer_count: int,
    ) -> None:
        """Log AFD payload sizing at the point it is split into P2P links."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        total_sender_bytes = sender_bytes_per_gpu * num_senders
        total_receiver_bytes = receiver_bytes_per_gpu * num_receivers
        scheduled_bytes = bytes_per_link * transfer_count
        remainder_per_sender = (
            sender_bytes_per_gpu - bytes_per_link * num_receivers
            if num_receivers > 0
            else 0
        )

        hidden_size = int(os.environ.get("AICONFIG_AFD_DEBUG_HIDDEN_SIZE", "0") or 0)
        dtype_bytes = int(os.environ.get("AICONFIG_AFD_DEBUG_DTYPE_BYTES", "2") or 2)
        token_equiv = None
        if hidden_size > 0 and dtype_bytes > 0:
            token_equiv = bytes_per_link / float(hidden_size * dtype_bytes)

        token_msg = (
            f", token_equiv_per_link={token_equiv:.3f} "
            f"(hidden={hidden_size}, dtype_bytes={dtype_bytes})"
            if token_equiv is not None
            else ""
        )
        logger.debug(
            "AFD payload split (%s): senders=%s receivers=%s "
            "sender_bytes_per_gpu=%s receiver_bytes_per_gpu=%s "
            "bytes_per_link=%s transfer_count=%s scheduled_bytes=%s "
            "total_sender_bytes=%s total_receiver_bytes=%s "
            "remainder_per_sender=%s%s",
            direction,
            num_senders,
            num_receivers,
            sender_bytes_per_gpu,
            receiver_bytes_per_gpu,
            bytes_per_link,
            transfer_count,
            scheduled_bytes,
            total_sender_bytes,
            total_receiver_bytes,
            remainder_per_sender,
            token_msg,
        )

    # ------------------------------------------------------------------
    # Multi-tier topology helpers (congestion-aware)
    # ------------------------------------------------------------------
    def _classify_tier(
        self, src_gpu: int, dst_gpu: int
    ) -> tuple[tuple, int, int]:
        """Classify a GPU-to-GPU transfer into a network tier.

        Returns ``(tier_key, src_id, dst_id)`` where:

        * *tier_key* = ``(tier_name, group_id)`` uniquely identifies an
          independent fabric domain. Distinct group IDs within the same
          tier have physically separate links, so they can't congest
          each other (e.g. NVLink inside node 0 vs node 1).
        * *src_id* / *dst_id* identify the communicating participants
          inside that fabric domain. For intra-node / intra-rack tiers
          these are already local topology indices. For inter-node /
          inter-rack tiers they are GPU-level NIC endpoint IDs and are
          densely remapped before the topology is built.

        Tier classification:

        * **intra-node** – both GPUs on the same node → NVLink topology.
          Group = node index.
        * **intra-rack** – same rack, different nodes (3-tier systems
          like GB200 with ``num_gpus_per_rack``) → NVSwitch topology.
          Group = rack index.
        * **inter-node** – different nodes, 2-tier system → IB fabric.
          Group = 0 (global inter-node fabric). Each GPU contributes a
          dedicated NIC endpoint, so a Switch topology built over the
          participating NICs models both source-NIC and destination-NIC
          contention for the specific source/destination pairs present
          in the transfer batch.
        * **inter-rack** – different racks (3-tier) → IB fabric.
          Group = 0 (global inter-rack fabric), using the same
          GPU-level NIC endpoint model as inter-node.
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
                # Inter-rack IB: use GPU NIC endpoints in one global fabric.
                return ("inter-rack", 0), src_gpu, dst_gpu

        # 2-tier system: inter-node IB, using GPU NIC endpoints.
        return ("inter-node", 0), src_gpu, dst_gpu

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
            topo = "Switch"
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

        # ── 1. Group transfers by fabric domain ─────────────────────
        tier_groups = self._group_tiered_transfers(transfers)

        if not tier_groups:
            return 0.0

        # ── 2. Simulate each group independently ────────────────────
        tier_latencies: dict[tuple, float] = {}

        for tier_key, group in tier_groups.items():
            tier_name, group_id = tier_key

            remapped_group, num_units, participant_to_local = (
                self._prepare_group_for_topology(tier_name, group)
            )

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

            for i, (local_src, local_dst, size_bytes) in enumerate(remapped_group):
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
                f"{len(group)} transfers across {len(participant_to_local)} participants "
                f"→ {latency_ms:.4f} ms"
            )

        # ── 3. Overall latency = slowest tier/group ─────────────────
        return max(tier_latencies.values()) if tier_latencies else 0.0

    def _group_tiered_transfers(
        self,
        transfers: list[tuple[int, int, int]],
    ) -> dict[tuple, list[tuple[int, int, int]]]:
        """Group transfers by fabric domain before simulation."""
        tier_groups: dict[tuple, list[tuple[int, int, int]]] = {}
        for src_gpu, dst_gpu, size_bytes in transfers:
            if src_gpu == dst_gpu:
                continue
            tier_key, src_id, dst_id = self._classify_tier(src_gpu, dst_gpu)
            logger.debug(
                f"Transfer group [{tier_key[0]}, group={tier_key[1]}]: "
                f"GPU {src_gpu} → GPU {dst_gpu}, {size_bytes} bytes"
            )
            tier_groups.setdefault(tier_key, []).append((src_id, dst_id, size_bytes))
        return tier_groups

    def _prepare_group_for_topology(
        self,
        tier_name: str,
        group: list[tuple[int, int, int]],
    ) -> tuple[list[tuple[int, int, int]], int, dict[int, int]]:
        """Prepare group for topology simulation using raw IDs.
        """
        max_local_id = max(max(src_id, dst_id) for src_id, dst_id, _ in group)
        participant_to_local = {
            participant_id: participant_id
            for src_id, dst_id, _ in group
            for participant_id in (src_id, dst_id)
        }
        return group, max_local_id + 1, participant_to_local

    def _build_worker_pp_transfer_plan(
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

    def _build_worker_dp_transfer_plan(
        self,
        src_attn_dp_pp_stages: list[list[list[int]]],
        dst_attn_dp_pp_stages: list[list[list[int]]],
        kv_cache_size: int,
        *,
        p_worker: int,
        d_worker: int,
    ) -> list[dict]:
        """Build a DP+PP+TP-aware KV transfer plan for one worker pair.

        ``kv_cache_size`` is interpreted as the full logical KV payload
        for the worker pair, before any DP/PP/TP sharding.  The payload
        is first partitioned across source/destination attention-DP
        replica pairs using ``lcm(src_dp, dst_dp)`` units, then each
        replica-pair payload is further split by PP stage and TP rank
        via ``_build_worker_pp_transfer_plan``.
        """
        if kv_cache_size <= 0:
            return []

        src_dp = len(src_attn_dp_pp_stages)
        dst_dp = len(dst_attn_dp_pp_stages)
        if src_dp == 0 or dst_dp == 0:
            return []

        dp_unit_count = math.lcm(src_dp, dst_dp)
        src_dp_units = dp_unit_count // src_dp
        dst_dp_units = dp_unit_count // dst_dp

        kv_per_dp_unit_base = kv_cache_size // dp_unit_count
        kv_per_dp_unit_rem = kv_cache_size % dp_unit_count

        dp_pair_bytes: dict[tuple[int, int], int] = {}
        
        # Balanced Partioning of KV across src DP replicas and dst DP replicas
        for dp_unit in range(dp_unit_count):
            unit_bytes = kv_per_dp_unit_base + (1 if dp_unit < kv_per_dp_unit_rem else 0)
            if unit_bytes == 0:
                continue
            src_dp_rank = dp_unit // src_dp_units
            dst_dp_rank = dp_unit // dst_dp_units
            key = (src_dp_rank, dst_dp_rank)
            dp_pair_bytes[key] = dp_pair_bytes.get(key, 0) + unit_bytes
        transfers: list[dict] = []

        for (src_dp_rank, dst_dp_rank), dp_kv_bytes in sorted(dp_pair_bytes.items()):
            pair_transfers = self._build_worker_pp_transfer_plan(
                src_attn_dp_pp_stages[src_dp_rank],
                dst_attn_dp_pp_stages[dst_dp_rank],
                dp_kv_bytes,
                p_worker=p_worker,
                d_worker=d_worker,
            )
            for transfer in pair_transfers:
                transfer["src_dp_rank"] = src_dp_rank
                transfer["dst_dp_rank"] = dst_dp_rank
            transfers.extend(pair_transfers)

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

        When worker layouts expose ``attn_dp_pp_stages``, KV bytes are
        first partitioned across attention-DP replicas and then across
        PP stages / TP ranks.  This supports both AFD workers
        (attention and FFN GPUs are physically separate) and shared-GPU
        MoE workers where ``attention_dp`` and ``moe_ep`` coexist on
        the same GPU pool.

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
                # Evenly ditribute the remainder kv_caches among the first few targets
                target_kv = kv_per_target_base + (
                    1 if target_idx < kv_per_target_rem else 0
                )
                prefill_worker = prefill_worker_layouts[p_idx]
                decode_worker = decode_worker_layouts[d_idx]
                p_dp_stages = prefill_worker.get("attn_dp_pp_stages") or [
                    prefill_worker["pp_stages"]
                ]
                d_dp_stages = decode_worker.get("attn_dp_pp_stages") or [
                    decode_worker["pp_stages"]
                ]
                pair_transfers = self._build_worker_dp_transfer_plan(
                    p_dp_stages,
                    d_dp_stages,
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
        gpu_ids: list[int] | None = None,
    ) -> float | None:
        """Simulate a collective communication operation.

        Resolution order:

        1. **Congestion-unaware multi-dim** – when both the
           congestion-unaware library and a ``system_spec`` are
           available, builds a native multi-dimensional topology
           (one dimension per network tier) and evaluates every
           GPU-to-GPU transfer via ``send()``.  No manual tier
           decomposition is needed; the C++ ``MultiDimTopology``
           routes each transfer through the correct dimension
           automatically.

        2. **Congestion-aware tiered** – if the congestion-unaware
           library is unavailable but a ``system_spec`` exists,
           falls back to the manual tier-classification path that
           creates per-tier 1-D congestion-aware topologies.

        3. **Congestion-aware flat** – no ``system_spec``: single
           flat topology (legacy path).

        Args:
            message_size_bytes: Total message payload in **bytes**.
            num_gpus: Number of GPUs participating.
            operation: One of ``"all_reduce"``, ``"all_gather"``,
                ``"reduce_scatter"``, ``"alltoall"``.
            gpu_ids: Explicit list of participating GPU IDs.  When
                ``None``, defaults to ``list(range(num_gpus))``
                (contiguous allocation).

        Returns:
            Simulated latency in **milliseconds**, or ``None`` on failure.
        """
        # ── Path 1: Congestion-unaware multi-dim ─────────────────
        if self.multidim_enabled and self._system_spec is not None:
            result = self.simulate_multidim_collective(
                message_size_bytes, num_gpus, operation, gpu_ids
            )
            if result is not None:
                return result
            # fall through on failure

        if not self._enabled:
            return None

        try:
            # ── Path 2: Congestion-aware tiered (manual tier split) ──
            if self._system_spec is not None:
                ids = gpu_ids if gpu_ids is not None else list(range(num_gpus))
                n = len(ids)
                transfers: list[tuple[int, int, int]] = []

                if operation in ("all_gather", "reduce_scatter", "alltoall"):
                    per_gpu_bytes = message_size_bytes // max(n, 1)
                    for src in ids:
                        for dst in ids:
                            if src != dst:
                                transfers.append((src, dst, per_gpu_bytes))
                elif operation == "all_reduce":
                    per_gpu_bytes = message_size_bytes // max(n, 1)
                    for src in ids:
                        for dst in ids:
                            if src != dst:
                                transfers.append((src, dst, per_gpu_bytes * 2))
                else:
                    raise ValueError(
                        f"Unsupported collective operation: {operation}"
                    )

                latency_ms = self._simulate_tiered_transfers(transfers)
                logger.debug(
                    f"AstraSim collective (tiered) {operation}: "
                    f"{message_size_bytes} bytes, {num_gpus} GPUs → "
                    f"{latency_ms:.4f} ms"
                )
                return latency_ms

            # ── Legacy single-topology path (no system_spec) ─────────
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

    def simulate_afd_transfers(
        self,
        attn_gpu_ids: list[int],
        ffn_gpu_ids: list[int],
        sender_bytes_per_gpu: int,
        receiver_bytes_per_gpu: int,
        pre_dispatch: bool = True,
    ) -> float | None:
        """Simulate AFD M-to-N transfers.

        Resolution order:

        1. **Congestion-unaware multi-dim** – uses native
           ``MultiDimTopology.send()`` for each attn↔FFN pair.
        2. **Congestion-aware tiered** – manual tier-classification
           with per-tier congestion modeling.

        Args:
            attn_gpu_ids: Attention GPU IDs.
            ffn_gpu_ids: FFN GPU IDs.
            sender_bytes_per_gpu: Bytes each sender GPU pushes.
            receiver_bytes_per_gpu: Bytes each receiver GPU pulls.
            pre_dispatch: ``True`` for attn→FFN (dispatch),
                ``False`` for FFN→attn (combine).

        Returns:
            Simulated latency in **milliseconds**, or ``None`` on failure.
        """
        if not attn_gpu_ids or not ffn_gpu_ids:
            return 0.0

        # ── Path 1: Congestion-unaware multi-dim ─────────────────
        if self.multidim_enabled and self._system_spec is not None:
            result = self.simulate_multidim_afd(
                attn_gpu_ids, ffn_gpu_ids,
                sender_bytes_per_gpu, receiver_bytes_per_gpu,
                pre_dispatch,
            )
            if result is not None:
                return result

        # ── Path 2: Congestion-aware tiered ──────────────────────
        if not self._enabled:
            return None

        try:
            transfers: list[tuple[int, int, int]] = []

            if pre_dispatch:
                # Attn → FFN: each attn GPU distributes sender_bytes
                # across all FFN GPUs.
                N = len(ffn_gpu_ids)
                bytes_per_link = max(1, sender_bytes_per_gpu // N) if sender_bytes_per_gpu > 0 else 0
                self._log_afd_payload_debug(
                    direction="attn->FFN",
                    num_senders=len(attn_gpu_ids),
                    num_receivers=N,
                    sender_bytes_per_gpu=sender_bytes_per_gpu,
                    receiver_bytes_per_gpu=receiver_bytes_per_gpu,
                    bytes_per_link=bytes_per_link,
                    transfer_count=len(attn_gpu_ids) * N,
                )
                if bytes_per_link > 0:
                    for attn_gpu in attn_gpu_ids:
                        for ffn_gpu in ffn_gpu_ids:
                            transfers.append((attn_gpu, ffn_gpu, bytes_per_link))
            else:
                # FFN → Attn: each FFN GPU distributes sender_bytes
                # across all attn GPUs.
                M = len(attn_gpu_ids)
                bytes_per_link = max(1, sender_bytes_per_gpu // M) if sender_bytes_per_gpu > 0 else 0
                self._log_afd_payload_debug(
                    direction="FFN->attn",
                    num_senders=len(ffn_gpu_ids),
                    num_receivers=M,
                    sender_bytes_per_gpu=sender_bytes_per_gpu,
                    receiver_bytes_per_gpu=receiver_bytes_per_gpu,
                    bytes_per_link=bytes_per_link,
                    transfer_count=len(ffn_gpu_ids) * M,
                )
                if bytes_per_link > 0:
                    for ffn_gpu in ffn_gpu_ids:
                        for attn_gpu in attn_gpu_ids:
                            transfers.append((ffn_gpu, attn_gpu, bytes_per_link))

            if not transfers:
                return 0.0

            latency_ms = self._simulate_tiered_transfers(transfers)
            direction = "attn→FFN" if pre_dispatch else "FFN→attn"
            logger.debug(
                f"AstraSim AFD ({direction}): "
                f"{len(attn_gpu_ids)} attn GPUs, {len(ffn_gpu_ids)} FFN GPUs, total {sender_bytes_per_gpu} bytes sent, {receiver_bytes_per_gpu} bytes received"
                f"{len(transfers)} transfers → {latency_ms:.4f} ms"
            )
            return latency_ms

        except Exception as e:
            logger.warning(f"AstraSim AFD transfer simulation failed: {e}")
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

            # Fall back to the preset ring.yaml topology
            # TODO: It should be safe to remove this as it should never be hit if no system_spec is input
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
