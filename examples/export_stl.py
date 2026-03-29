"""Export a wing design to STEP, IGES, STL, and BREP formats.

Demonstrates the NURBS export pipeline using native OCC writers.
STEP and IGES preserve exact NURBS geometry; STL is tessellated.
"""

import os

from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    VolumeCalculator,
    NurbsExporter,
)
from aeroshape import MeshTopologyManager

EXPORT_DIR = "Exports"


def main():
    # Define wing
    root = AirfoilProfile.from_naca4("2412", num_points=50)
    tip = AirfoilProfile.from_naca4("0012", num_points=50)

    wing = MultiSegmentWing(name="Export Wing")
    wing.add_segment(SegmentSpec(
        span=8.0,
        root_airfoil=root,
        tip_airfoil=tip,
        root_chord=2.5,
        tip_chord=0.8,
        sweep_le_deg=20.0,
        num_sections=20,
    ))

    # Compute properties for reference
    X, Y, Z = wing.to_vertex_grids(num_points_profile=50)
    triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
    volume = VolumeCalculator.compute_solid_volume(triangles)
    mass = volume * 2700.0

    print(f"Wing volume: {volume:.6f} m^3")
    print(f"Wing mass (aluminum): {mass:.2f} kg")

    # Build NURBS shape
    shape = wing.to_occ_shape()

    # Export to all formats
    os.makedirs(EXPORT_DIR, exist_ok=True)

    step = os.path.join(EXPORT_DIR, "wing_export.step")
    NurbsExporter.to_step(shape, step)
    print(f"STEP exported: {step}")

    iges = os.path.join(EXPORT_DIR, "wing_export.iges")
    NurbsExporter.to_iges(shape, iges)
    print(f"IGES exported: {iges}")

    stl = os.path.join(EXPORT_DIR, "wing_export.stl")
    NurbsExporter.to_stl(shape, stl, linear_deflection=0.01)
    print(f"STL exported:  {stl}")

    brep = os.path.join(EXPORT_DIR, "wing_export.brep")
    NurbsExporter.to_brep(shape, brep)
    print(f"BREP exported: {brep}")


if __name__ == "__main__":
    main()
