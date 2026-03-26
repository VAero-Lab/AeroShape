"""AeroShape: CAD-free volume and mass properties computation for lifting surfaces.

An open-source Python implementation of the GVM (Geometry, Volume, and Mass)
methodology for computing volume, mass, center of gravity, and moments of
inertia of 3D lifting surfaces and wing-box structures.

Reference:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378.
"""

from aeroshape.geometry import NACAProfileGenerator, WingMeshFactory
from aeroshape.mesh_utils import MeshTopologyManager
from aeroshape.volume import VolumeCalculator
from aeroshape.mass import MassPropertiesCalculator
from aeroshape.exporter import ModelExporter
from aeroshape.visualization import show_interactive, show_static

__version__ = "2.0.0"

__all__ = [
    "NACAProfileGenerator",
    "WingMeshFactory",
    "MeshTopologyManager",
    "VolumeCalculator",
    "MassPropertiesCalculator",
    "ModelExporter",
    "show_interactive",
    "show_static",
]
