"""
AstraSim network simulator utilities.
Centralized import and initialization to be shared across modules.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Get paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

# Default network config path
DEFAULT_NETWORK_CONFIG = os.path.join(
    _PROJECT_ROOT, "network_backend", "astra-network-analytical", "input", "Ring.yml"
)

# Network simulator library path
_NETWORK_SIM_LIB_PATH = os.path.join(
    _PROJECT_ROOT, "network_backend", "astra-network-analytical", "lib"
)

# Define exposed params from the python bindings and will be used by other python files
NETWORK_SIM_AVAILABLE = False
EventQueue = None
Topology = None
Chunk = None
NetworkParser = None
construct_topology = None

# Add library path to sys.path if not already there
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


def is_available() -> bool:
    """Check if AstraSim is available."""
    return NETWORK_SIM_AVAILABLE


def get_default_config() -> str:
    """Get default network configuration file path."""
    return DEFAULT_NETWORK_CONFIG