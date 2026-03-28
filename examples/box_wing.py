"""Box-Wing (Prandtl-plane) configuration.

Models a box-wing with two lifting surfaces at different heights
connected at the tips by a near-vertical fin, forming a closed
rectangular lifting system when viewed from the front.

Based on Prandtl's "best wing system" (1924) and modern Prandtl-plane
concepts (Frediani et al.). Similar to configurations in
Gagnon & Zingg (AIAA 2013-2850, Fig. 8).

Uses AircraftModel to assemble:
- Lower (front) wing with an upward tip connector (high-dihedral fin)
- Upper (rear) wing: aft, elevated, forward-swept so its tip meets the
  top of the connector
"""

import math
from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    AircraftModel,
    VolumeCalculator,
    MassPropertiesCalculator,
    NurbsExporter,
    show_interactive,
)
from aeroshape.mesh_utils import MeshTopologyManager


def main():
    # ── Airfoil profiles ──────────────────────────────────────────
    root_profile = AirfoilProfile.from_naca4("2412", num_points=50)
    tip_profile = AirfoilProfile.from_naca4("0010", num_points=50)
    fin_profile = AirfoilProfile.from_naca4("0008", num_points=50)

    # ── Planform parameters ──────────────────────────────────────
    semi_span = 10.0          # semi-span of each main wing [m]
    lower_sweep_deg = 5.0     # lower wing LE sweep [deg]
    vertical_gap = 3.0        # vertical separation between wings [m]
    connector_dy = 0.5        # spanwise extent of tip connector [m]
    x_upper_origin = 4.0      # upper wing root aft offset [m]

    # ── Derived angles ───────────────────────────────────────────
    # Connector dihedral to rise `vertical_gap` over `connector_dy`
    connector_dihedral_deg = math.degrees(
        math.atan(vertical_gap / connector_dy)
    )

    # Lower wing tip X (from sweep)
    lower_tip_x = semi_span * math.tan(math.radians(lower_sweep_deg))

    # Upper wing forward-sweep angle so its tip X matches the connector top
    connector_top_x = lower_tip_x        # connector has 0 deg sweep
    connector_top_y = semi_span + connector_dy
    upper_sweep_deg = math.degrees(
        math.atan((connector_top_x - x_upper_origin) / connector_top_y)
    )

    # ── Lower (front) wing + vertical tip connector ──────────────
    lower_wing = MultiSegmentWing(name="Lower Wing")

    # Main wing section (flat)
    lower_wing.add_segment(SegmentSpec(
        span=semi_span,
        root_airfoil=root_profile,
        tip_airfoil=tip_profile,
        root_chord=3.0,
        tip_chord=1.8,
        sweep_le_deg=lower_sweep_deg,
        dihedral_deg=0.0,
        twist_deg=-1.0,
        num_sections=12,
    ))

    # Vertical tip connector (high-dihedral fin)
    lower_wing.add_segment(SegmentSpec(
        span=connector_dy,
        root_airfoil=tip_profile,
        tip_airfoil=fin_profile,
        root_chord=1.8,
        tip_chord=1.2,
        sweep_le_deg=0.0,
        dihedral_deg=connector_dihedral_deg,
        num_sections=6,
    ))

    # ── Upper (rear) wing ────────────────────────────────────────
    # Forward-swept so tip meets the connector top
    upper_wing = MultiSegmentWing(name="Upper Wing")
    upper_wing.add_segment(SegmentSpec(
        span=connector_top_y,
        root_airfoil=root_profile,
        tip_airfoil=fin_profile,
        root_chord=2.5,
        tip_chord=1.2,
        sweep_le_deg=upper_sweep_deg,
        dihedral_deg=0.0,
        twist_deg=1.0,
        num_sections=14,
    ))

    # ── Aircraft assembly ────────────────────────────────────────
    aircraft = AircraftModel(name="Box Wing")
    aircraft.add_surface(lower_wing, origin=(0.0, 0.0, 0.0))
    aircraft.add_surface(upper_wing, origin=(x_upper_origin, 0.0, vertical_gap))

    # Print tip positions for verification
    lower_tip = (lower_tip_x, semi_span + connector_dy,
                 vertical_gap)
    upper_tip = (x_upper_origin + connector_top_y
                 * math.tan(math.radians(upper_sweep_deg)),
                 connector_top_y, vertical_gap)
    print(f"Lower wing connector top: "
          f"({lower_tip[0]:.3f}, {lower_tip[1]:.1f}, {lower_tip[2]:.1f})")
    print(f"Upper wing tip:           "
          f"({upper_tip[0]:.3f}, {upper_tip[1]:.1f}, {upper_tip[2]:.1f})")

    # ── GVM Analysis ─────────────────────────────────────────────
    triangles = aircraft.to_triangles(num_points_profile=50)
    volume = VolumeCalculator.compute_solid_volume(triangles)
    density = 2700.0
    mass = volume * density

    grids = aircraft.to_vertex_grids_list(num_points_profile=50)
    print(f"\nConfiguration: {aircraft.name}")
    print(f"  Surfaces: {len(aircraft.surfaces)}")
    for X, Y, Z, name in grids:
        tris = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
        v = VolumeCalculator.compute_solid_volume(tris)
        m = v * density
        cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, m)
        print(f"    {name}:")
        print(f"      Volume: {v:.6f} m^3, Mass: {m:.1f} kg")
        print(f"      CG: ({cg[0]:.3f}, {cg[1]:.3f}, {cg[2]:.3f})")

    print(f"\n  Total volume: {volume:.4f} m^3")
    print(f"  Total mass:   {mass:.1f} kg")

    # ── NURBS export ─────────────────────────────────────────────
    shape = aircraft.to_occ_shape(fuse=False)
    NurbsExporter.to_step(shape, "box_wing.step")
    print("\n  STEP exported: box_wing.step")

    # ── Visualize ────────────────────────────────────────────────
    cg = (0, 0, 0)
    inertia = (0, 0, 0, 0, 0, 0)
    show_interactive(triangles, volume, mass, cg, inertia,
                     title=aircraft.name)


if __name__ == "__main__":
    main()
