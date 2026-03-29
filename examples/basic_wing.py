"""Basic solid wing volume and mass computation.

Demonstrates the core AeroShape workflow using the NURBS pipeline:
1. Define airfoil profiles
2. Build a multi-segment wing
3. Compute volume via the Divergence Theorem
4. Compute mass properties (center of mass and inertia tensor)
5. Build NURBS shape and export to STEP

Reference:
    Valencia et al., Aerospace Science and Technology 108 (2021) 106378.
"""

import os

from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    VolumeCalculator,
    MassPropertiesCalculator,
    NurbsExporter,
    show_interactive,
)
from aeroshape import MeshTopologyManager

EXPORT_DIR = "Exports"


def main():
    # Step 1: Define airfoil profiles
    root_profile = AirfoilProfile.from_naca4("2412", num_points=40)
    tip_profile = AirfoilProfile.from_naca4("2412", num_points=40)

    # Step 2: Build wing
    wing = MultiSegmentWing(name="NACA 2412 Wing")
    wing.add_segment(SegmentSpec(
        span=10.0,
        root_airfoil=root_profile,
        tip_airfoil=tip_profile,
        root_chord=2.0,
        tip_chord=1.0,
        sweep_le_deg=15.0,
        num_sections=15,
    ))

    # Step 3: Get vertex grids and triangulate
    X, Y, Z = wing.to_vertex_grids(num_points_profile=40)
    print(f"Vertex grid: {X.shape[0]} sections x {X.shape[1]} points")

    triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
    print(f"Total triangles: {len(triangles)}")

    # Step 4: Volume computation via Divergence Theorem
    volume = VolumeCalculator.compute_solid_volume(triangles)
    print(f"\nVolume (Divergence Theorem): {volume:.6f} m^3")

    # Step 5: Mass properties (EPS foam, rho = 50 kg/m^3)
    density = 50.0
    mass = volume * density
    cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = inertia

    print(f"Mass (EPS foam): {mass:.4f} kg")
    print(f"\nCenter of Mass:")
    print(f"  X (chord):  {cg[0]:.5f} m")
    print(f"  Y (span):   {cg[1]:.5f} m")
    print(f"  Z (height): {cg[2]:.5f} m")
    print(f"\nMoments of Inertia:")
    print(f"  Ixx = {Ixx:.5f} kg*m^2")
    print(f"  Iyy = {Iyy:.5f} kg*m^2")
    print(f"  Izz = {Izz:.5f} kg*m^2")

    # Step 6: NURBS export
    os.makedirs(EXPORT_DIR, exist_ok=True)
    shape = wing.to_occ_shape()
    step_path = os.path.join(EXPORT_DIR, "basic_wing.step")
    NurbsExporter.to_step(shape, step_path)
    print(f"\nSTEP exported: {step_path}")

    # Step 7: Visualization
    show_interactive(
        triangles, volume, mass, cg, inertia,
        title="AeroShape - NACA 2412 Solid Wing",
    )


if __name__ == "__main__":
    main()
