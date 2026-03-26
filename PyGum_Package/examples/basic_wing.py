"""Basic solid wing volume and mass computation.

This example demonstrates the core GVM workflow:
1. Generate a NACA-based wing mesh
2. Triangulate the surface
3. Compute volume via the Divergence Theorem (Eq. 2)
4. Compute mass properties (center of mass and inertia tensor)

Reference:
    Valencia et al., Aerospace Science and Technology 108 (2021) 106378.
"""

from aeroshape import (
    WingMeshFactory,
    MeshTopologyManager,
    VolumeCalculator,
    MassPropertiesCalculator,
    show_interactive,
    show_static,
)


def main():
    # Wing parameters
    naca_root = "2412"
    naca_tip = "2412"
    semi_span = 10.0       # meters
    chord_root = 2.0       # meters
    chord_tip = 1.0        # meters
    sweep_angle = 15.0     # degrees
    num_points = 40        # chordwise points per profile
    num_sections = 15      # spanwise sections

    # Step 1: Generate the 3D wing mesh (vertex representation Vk)
    print("Generating wing mesh...")
    X, Y, Z = WingMeshFactory.create(
        naca_root, naca_tip, semi_span, chord_root, chord_tip,
        sweep_angle, num_points, num_sections
    )
    print(f"  Vertex grid: {X.shape[0]} sections x {X.shape[1]} points")

    # Step 2: Triangulate the outer surface (watertight for volume)
    print("Triangulating surface...")
    triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
    print(f"  Total triangles: {len(triangles)}")

    # Step 3: Volume computation via Divergence Theorem
    volume = VolumeCalculator.compute_solid_volume(triangles)
    print(f"\nVolume (Divergence Theorem): {volume:.6f} m^3")

    # Step 4: Mass properties (using EPS foam, rho = 50 kg/m^3)
    density = 50.0  # kg/m^3
    mass = volume * density
    print(f"Mass (EPS foam): {mass:.4f} kg")

    cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = inertia

    print(f"\nCenter of Mass:")
    print(f"  X (chord):  {cg[0]:.5f} m")
    print(f"  Y (span):   {cg[1]:.5f} m")
    print(f"  Z (height): {cg[2]:.5f} m")

    print(f"\nMoments of Inertia:")
    print(f"  Ixx = {Ixx:.5f} kg*m^2")
    print(f"  Iyy = {Iyy:.5f} kg*m^2")
    print(f"  Izz = {Izz:.5f} kg*m^2")
    print(f"  Ixy = {Ixy:.5f} kg*m^2")
    print(f"  Ixz = {Ixz:.5f} kg*m^2")
    print(f"  Iyz = {Iyz:.5f} kg*m^2")

    # Step 5: Visualization
    show_interactive(
        triangles, volume, mass, cg, inertia,
        title="AeroShape - NACA 2412 Solid Wing",
    )


if __name__ == "__main__":
    main()
