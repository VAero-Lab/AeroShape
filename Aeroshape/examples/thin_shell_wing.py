"""Thin-shell wing analysis comparing both volume approaches.

Demonstrates both thin-shell volume computation methods from the paper:
- Approach I (Offset): exact via Divergence Theorem on double-walled mesh
- Approach II (Unfolding): approximation via wetted_area * t_shell

Both approaches are now part of the aeroshape.volume module.

Reference:
    Valencia et al., Aerospace Science and Technology 108 (2021) 106378,
    Section 2.2.1 "Volume computation as thin-shell".
"""

from aeroshape import (
    WingMeshFactory,
    MeshTopologyManager,
    VolumeCalculator,
    MassPropertiesCalculator,
    show_interactive,
)


def main():
    # Wing parameters (same as convergence study in paper, Section 3.1.2)
    naca_root = "2412"
    naca_tip = "2412"
    semi_span = 10.0
    chord_root = 2.0
    chord_tip = 1.0
    sweep_angle = 15.0
    num_points = 100      # fine chordwise resolution
    num_sections = 15
    t_shell = 0.001       # 1 mm aluminum sheet

    # Generate wing mesh
    X, Y, Z = WingMeshFactory.create(
        naca_root, naca_tip, semi_span, chord_root, chord_tip,
        sweep_angle, num_points, num_sections
    )

    # --- Approach I: Offset method (exact, Divergence Theorem) ---
    print("=== Approach I: Offset Method ===")
    volume_exact, vol_outer, vol_inner = (
        VolumeCalculator.compute_shell_volume_offset(X, Y, Z, t_shell)
    )
    print(f"Outer volume:  {vol_outer:.8f} m^3")
    print(f"Inner volume:  {vol_inner:.8f} m^3")
    print(f"Shell volume (V_outer - V_inner): {volume_exact:.8f} m^3")

    # --- Approach II: Unfolding method (area x thickness) ---
    print("\n=== Approach II: Unfolding Method ===")
    solid_triangles = MeshTopologyManager.get_wing_triangles(
        X, Y, Z, closed=True
    )
    wetted_area = VolumeCalculator.compute_surface_area(solid_triangles)
    volume_approx = VolumeCalculator.compute_shell_volume_unfolding(
        solid_triangles, t_shell
    )
    print(f"Wetted area: {wetted_area:.6f} m^2")
    print(f"Shell volume (Area x t): {volume_approx:.8f} m^3")

    # --- Comparison ---
    error = abs(volume_approx - volume_exact)
    error_pct = error / volume_exact * 100 if volume_exact > 0 else 0
    print(f"\n=== Comparison ===")
    print(f"Absolute difference: {error:.8f} m^3")
    print(f"Relative difference: {error_pct:.4f}%")

    # --- Mass properties (aluminum, rho = 2810 kg/m^3) ---
    density = 2810.0
    mass = volume_exact * density
    cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

    print(f"\nMass (aluminum shell): {mass:.4f} kg")
    print(f"Center of Mass: X={cg[0]:.5f}, Y={cg[1]:.5f}, Z={cg[2]:.5f}")

    # Visualization
    show_interactive(
        solid_triangles, volume_exact, mass, cg, inertia,
        title="AeroShape - Thin Shell Wing (1 mm aluminum)",
    )


if __name__ == "__main__":
    main()
