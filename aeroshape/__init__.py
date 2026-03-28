"""AeroShape: NURBS-based geometry engine for aircraft design and analysis.

Provides scriptable 3D modeling of conventional and unconventional aircraft
with NURBS-based geometry representations. Includes the GVM (Geometry,
Volume, and Mass) methodology for computing volume, mass, center of gravity,
and moments of inertia of lifting surfaces.

References:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378.
"""

# Core GVM pipeline (always available — numpy only)
from aeroshape.mesh_utils import MeshTopologyManager
from aeroshape.volume import VolumeCalculator
from aeroshape.mass import MassPropertiesCalculator
from aeroshape.visualization import show_interactive, show_static

# NURBS geometry engine
from aeroshape.profiles import AirfoilProfile
from aeroshape.wing_model import SegmentSpec, MultiSegmentWing, AircraftModel
from aeroshape.nurbs_ops import (
    NurbsSurfaceBuilder, tessellate_shape, sample_shape_grid,
    occ_mass_properties, make_wire_from_points, make_line_wire,
)
from aeroshape.nurbs_export import NurbsExporter

__version__ = "3.0.0"

__all__ = [
    # GVM pipeline
    "MeshTopologyManager",
    "VolumeCalculator",
    "MassPropertiesCalculator",
    "show_interactive",
    "show_static",
    # Geometry engine
    "AirfoilProfile",
    "SegmentSpec",
    "MultiSegmentWing",
    "AircraftModel",
    # NURBS operations
    "NurbsSurfaceBuilder",
    "NurbsExporter",
    "tessellate_shape",
    "sample_shape_grid",
    "occ_mass_properties",
    "make_wire_from_points",
    "make_line_wire",
]
