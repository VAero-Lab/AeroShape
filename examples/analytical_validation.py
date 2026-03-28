"""Analytical validation using a unit cube (Section 3.1.1 of the paper).

Validates the Divergence Theorem implementation by computing the
volume of a 1x1x1 unit cube, whose analytical volume is 1.0 m^3.

The Divergence Theorem requires consistently-oriented normals (all
outward or all inward). This test uses outward-pointing normals.

This mirrors the validation approach in Section 3.1.1 (Fig. 6, Table 2).

Reference:
    Valencia et al., Aerospace Science and Technology 108 (2021) 106378.
"""

import numpy as np
from aeroshape import (
    VolumeCalculator,
    MassPropertiesCalculator,
    MeshTopologyManager,
)
from aeroshape.geometry import WingMeshFactory


def test_cube_volume():
    """Validate volume computation with a unit cube (analytical proof)."""
    print("=" * 60)
    print("  TEST 1: Unit Cube Volume (Section 3.1.1, Table 2)")
    print("=" * 60)

    # 8 vertices of a unit cube
    P0 = np.array([0, 0, 0], dtype=float)
    P1 = np.array([1, 0, 0], dtype=float)
    P2 = np.array([1, 1, 0], dtype=float)
    P3 = np.array([0, 1, 0], dtype=float)
    P4 = np.array([0, 0, 1], dtype=float)
    P5 = np.array([1, 0, 1], dtype=float)
    P6 = np.array([1, 1, 1], dtype=float)
    P7 = np.array([0, 1, 1], dtype=float)

    # 12 triangles with consistent outward-pointing normals
    # (counter-clockwise winding when viewed from outside)
    triangles = [
        # Bottom face (Z=0), normal = [0,0,-1]
        (P0, P2, P1), (P0, P3, P2),
        # Top face (Z=1), normal = [0,0,+1]
        (P4, P5, P6), (P4, P6, P7),
        # Front face (Y=0), normal = [0,-1,0]
        (P0, P1, P5), (P0, P5, P4),
        # Back face (Y=1), normal = [0,+1,0]
        (P2, P3, P7), (P2, P7, P6),
        # Right face (X=1), normal = [+1,0,0]
        (P1, P2, P6), (P1, P6, P5),
        # Left face (X=0), normal = [-1,0,0]
        (P0, P7, P3), (P0, P4, P7),
    ]

    volume = VolumeCalculator.compute_solid_volume(triangles)

    print(f"\nTriangles: {len(triangles)}")
    print(f"Expected volume: 1.000000 m^3")
    print(f"Computed volume: {volume:.6f} m^3")
    print(f"Error: {abs(volume - 1.0):.2e}")

    vol_ok = abs(volume - 1.0) < 1e-10
    print(f"Result: {'PASS' if vol_ok else 'FAIL'}")
    return vol_ok


def test_wing_volume_convergence():
    """Test volume convergence with increasing mesh resolution.

    Mirrors the convergence study in Section 3.1.2 (Fig. 7).
    """
    print("\n" + "=" * 60)
    print("  TEST 2: Wing Volume Convergence (Section 3.1.2)")
    print("=" * 60)

    resolutions = [10, 20, 40, 80, 100]
    volumes = []

    for n_pts in resolutions:
        X, Y, Z = WingMeshFactory.create(
            naca_root="2412", naca_tip="2412",
            semi_span=1.0, chord_root=0.25, chord_tip=0.15,
            sweep_angle_deg=0.0, num_points_profile=n_pts, num_sections=4
        )
        tris = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
        vol = VolumeCalculator.compute_solid_volume(tris)
        volumes.append(vol)
        print(f"  Points={n_pts:3d}  Triangles={len(tris):5d}  "
              f"Volume={vol:.8f} m^3")

    # Check convergence: last two resolutions should be very close
    rel_change = abs(volumes[-1] - volumes[-2]) / volumes[-1] * 100
    print(f"\nRelative change (last two): {rel_change:.4f}%")
    conv_ok = rel_change < 1.0
    print(f"Convergence: {'PASS' if conv_ok else 'FAIL'} "
          f"(threshold: <1%)")
    return conv_ok


def test_mass_distribution():
    """Test mass distribution model (Section 2.3.1).

    For a symmetric untapered wing, the center of mass should be
    at the mid-span and near mid-chord.
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Mass Distribution Model (Section 2.3.1)")
    print("=" * 60)

    X, Y, Z = WingMeshFactory.create(
        naca_root="0012", naca_tip="0012",
        semi_span=10.0, chord_root=2.0, chord_tip=2.0,
        sweep_angle_deg=0.0, num_points_profile=50, num_sections=10
    )

    tris = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
    volume = VolumeCalculator.compute_solid_volume(tris)
    mass = volume * 1000.0  # arbitrary density

    cg, inertia, M_3D = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

    print(f"\nSymmetric wing NACA 0012, untapered, unswept")
    print(f"Volume: {volume:.6f} m^3, Mass: {mass:.2f} kg")
    print(f"Center of Mass: X={cg[0]:.4f}, Y={cg[1]:.4f}, Z={cg[2]:.4f}")

    # For symmetric NACA 0012 (no camber), CG_Z should be ~0
    # CG_Y should be near mid-span for untapered wing
    # CG_X should be near mid-chord
    z_ok = abs(cg[2]) < 0.01
    y_ok = abs(cg[1] - 5.0) < 1.0  # near mid-span
    x_ok = 0 < cg[0] < 2.0  # within chord range

    print(f"\nZ symmetry (|CG_Z| < 0.01): {'PASS' if z_ok else 'FAIL'} "
          f"(CG_Z = {cg[2]:.6f})")
    print(f"Y mid-span (CG_Y ~ 5.0): {'PASS' if y_ok else 'FAIL'}")
    print(f"X within chord (0 < CG_X < 2): {'PASS' if x_ok else 'FAIL'}")

    # Verify mass conservation
    mass_sum = np.sum(M_3D)
    mass_ok = abs(mass_sum - mass) / mass < 0.01
    print(f"Mass conservation: {'PASS' if mass_ok else 'FAIL'} "
          f"(sum={mass_sum:.4f}, total={mass:.4f})")

    return z_ok and y_ok and x_ok and mass_ok


def main():
    results = [
        test_cube_volume(),
        test_wing_volume_convergence(),
        test_mass_distribution(),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"  SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
