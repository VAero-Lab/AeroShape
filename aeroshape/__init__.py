"""AeroShape: NURBS-based geometry engine for aircraft design and analysis.

Provides scriptable 3D modeling of conventional and unconventional aircraft
with NURBS-based geometry representations. Includes the GVM (Geometry,
Volume, and Mass) methodology for computing volume, mass, center of gravity,
and moments of inertia of lifting surfaces.

Package structure:
    aeroshape.core         — GVM pipeline: mesh, volume, mass, clustering
    aeroshape.geometry     — Component definitions: airfoils, wings, aircraft
    aeroshape.cad          — NURBS/CAD operations and export
    aeroshape.vis          — Visualization and rendering

References:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378.
"""

# Core GVM pipeline
from aeroshape.core.mesh import MeshTopologyManager
from aeroshape.core.volume import VolumeCalculator
from aeroshape.core.mass import MassPropertiesCalculator
from aeroshape.core import clustering

# Visualization
from aeroshape.vis.rendering import show_interactive, show_static

# Geometry engine
from aeroshape.geometry.airfoils import AirfoilProfile, NACAProfileGenerator
from aeroshape.geometry.wings import SegmentSpec, MultiSegmentWing
from aeroshape.geometry.aircraft import AircraftModel

# CAD / NURBS operations
from aeroshape.cad.surfaces import NurbsSurfaceBuilder
from aeroshape.cad.export import NurbsExporter
from aeroshape.cad.utils import (
    tessellate_shape, sample_shape_grid, occ_mass_properties,
    make_wire_from_points, make_line_wire,
)

__version__ = "4.0.0"

__all__ = [
    # Core GVM pipeline
    "MeshTopologyManager",
    "VolumeCalculator",
    "MassPropertiesCalculator",
    "clustering",
    # Visualization
    "show_interactive",
    "show_static",
    # Geometry engine
    "AirfoilProfile",
    "NACAProfileGenerator",
    "SegmentSpec",
    "MultiSegmentWing",
    "AircraftModel",
    # CAD / NURBS operations
    "NurbsSurfaceBuilder",
    "NurbsExporter",
    "tessellate_shape",
    "sample_shape_grid",
    "occ_mass_properties",
    "make_wire_from_points",
    "make_line_wire",
]
