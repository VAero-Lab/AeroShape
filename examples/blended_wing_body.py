"""Blended Wing Body (BWB) configuration.

Models a BWB-like planform with a thick center body that smoothly
transitions into thin outboard wings, similar to the configurations
described in Gagnon & Zingg (AIAA 2013-2850, Fig. 6) and Liebeck's
BWB concept.

The BWB is modeled as a single multi-segment wing with:
- Center body: thick symmetric airfoil, large chord
- Transition: blending from thick body to conventional airfoil
- Outboard wing: thin airfoil, moderate sweep
- Wingtip: tapered to thin tip
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
    # ── Airfoil profiles ──────────────────────────────────────────
    # Center body: thick symmetric (like a 30% thick section)
    center = AirfoilProfile.from_naca4("0030", num_points=60)
    # Transition: moderately thick cambered
    transition = AirfoilProfile.from_naca4("4418", num_points=60)
    # Outboard: conventional transonic
    outboard = AirfoilProfile.from_naca4("2412", num_points=60)
    # Tip: thin symmetric
    tip = AirfoilProfile.from_naca4("0009", num_points=60)

    # ── Multi-segment BWB ─────────────────────────────────────────
    bwb = MultiSegmentWing(name="Blended Wing Body")

    # Center body (half-span = 3m, large chord, no sweep)
    bwb.add_segment(SegmentSpec(
        span=3.0,
        root_airfoil=center,
        tip_airfoil=transition,
        root_chord=12.0,
        tip_chord=6.0,
        sweep_le_deg=40.0,
        dihedral_deg=0.0,
        twist_deg=0.0,
        num_sections=10,
    ))

    # Transition to outboard wing
    bwb.add_segment(SegmentSpec(
        span=3.0,
        root_airfoil=transition,
        tip_airfoil=outboard,
        root_chord=6.0,
        tip_chord=3.0,
        sweep_le_deg=35.0,
        dihedral_deg=2.0,
        twist_deg=-1.0,
        num_sections=8,
    ))

    # Outboard wing
    bwb.add_segment(SegmentSpec(
        span=8.0,
        root_airfoil=outboard,
        tip_airfoil=tip,
        root_chord=3.0,
        tip_chord=1.0,
        sweep_le_deg=30.0,
        dihedral_deg=4.0,
        twist_deg=-3.0,
        num_sections=12,
    ))

    # Wingtip
    bwb.add_segment(SegmentSpec(
        span=0.5,
        root_airfoil=tip,
        tip_airfoil=tip,
        root_chord=1.0,
        tip_chord=0.4,
        sweep_le_deg=45.0,
        dihedral_deg=10.0,
        num_sections=4,
    ))

    # ── GVM Analysis ──────────────────────────────────────────────
    num_pts = 60
    X, Y, Z = bwb.to_vertex_grids(num_points_profile=num_pts)
    triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
    volume = VolumeCalculator.compute_solid_volume(triangles)

    density = 150.0  # composite structure, kg/m^3
    mass = volume * density
    cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

    print(f"Configuration: {bwb.name}")
    print(f"  Segments: {len(bwb.segments)}")
    print(f"  Total span (half): {sum(s.span for s in bwb.segments):.1f} m")
    print(f"  Volume: {volume:.4f} m^3")
    print(f"  Mass:   {mass:.1f} kg")
    print(f"  CG:     ({cg[0]:.3f}, {cg[1]:.3f}, {cg[2]:.3f}) m")
    print(f"  Ixx={inertia[0]:.1f}, Iyy={inertia[1]:.1f}, "
          f"Izz={inertia[2]:.1f} kg*m^2")

    # ── NURBS export ──────────────────────────────────────────────
    os.makedirs(EXPORT_DIR, exist_ok=True)
    shape = bwb.to_occ_shape()
    step_path = os.path.join(EXPORT_DIR, "blended_wing_body.step")
    NurbsExporter.to_step(shape, step_path)
    print(f"\n  STEP exported: {step_path}")

    # ── Visualize ─────────────────────────────────────────────────
    show_interactive(triangles, volume, mass, cg, inertia, title=bwb.name)


if __name__ == "__main__":
    main()
