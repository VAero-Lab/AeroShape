"""Blended Wing Body (BWB) with leading- and trailing-edge guide curves.

Instead of defining discrete segments (which produce linear LE/TE lines
with kinks at segment boundaries), this example defines the BWB planform
via smooth B-spline guide curves for the leading and trailing edges.

MultiSegmentWing.from_planform_curves() fits B-spline curves through the
LE and TE control points, samples them at many intermediate stations, and
lofts through all sections to produce a smooth surface with no kinks.

Key benefits over the segment-based approach:
- Smooth LE and TE transitions (C2-continuous curves)
- No visible breaks between regions (center body, transition, outboard)
- Guide curves give direct control over the planform shape
"""

import os

from aeroshape import (
    AirfoilProfile,
    MultiSegmentWing,
    VolumeCalculator,
    MassPropertiesCalculator,
    NurbsExporter,
    show_interactive,
)
from aeroshape import MeshTopologyManager

EXPORT_DIR = "Exports"


def main():
    # ── Airfoil profiles at key spanwise stations ────────────────
    # Center body: very thick symmetric (cabin volume)
    center = AirfoilProfile.from_naca4("0025", num_points=60)
    # Inner transition: thick cambered
    transition = AirfoilProfile.from_naca4("4418", num_points=60)
    # Outboard root: conventional
    outboard = AirfoilProfile.from_naca4("2412", num_points=60)
    # Outboard mid: thinner
    mid_outboard = AirfoilProfile.from_naca4("2410", num_points=60)
    # Tip: thin symmetric
    tip = AirfoilProfile.from_naca4("0009", num_points=60)

    # ── Leading-edge guide curve ─────────────────────────────────
    # Smooth B-spline fitted through these points (X=chord, Y=span)
    le_points = [
        (0.0,  0.0,  0.0),     # center body LE
        (0.8,  1.5,  0.0),     # inner body LE
        (2.0,  3.0,  0.0),     # transition start LE
        (4.0,  6.0,  0.0),     # transition end LE
        (6.5,  10.0, 0.05),    # outboard LE (slight rise)
        (8.5,  13.0, 0.15),    # near-tip LE
        (9.5,  14.5, 0.25),    # tip LE
    ]

    # ── Trailing-edge guide curve ────────────────────────────────
    te_points = [
        (12.0, 0.0,  0.0),     # center body TE (chord ~ 12 m)
        (11.0, 1.5,  0.0),     # inner body TE
        (9.5,  3.0,  0.0),     # transition TE (chord ~ 7.5 m)
        (8.5,  6.0,  0.0),     # transition end TE (chord ~ 4.5 m)
        (9.5,  10.0, 0.05),    # outboard TE (chord ~ 3 m)
        (10.0, 13.0, 0.15),    # near-tip TE (chord ~ 1.5 m)
        (10.0, 14.5, 0.25),    # tip TE (chord ~ 0.5 m)
    ]

    # ── Airfoil stations (spanwise fraction, profile) ────────────
    # Fraction 0 = root (Y=0), fraction 1 = tip (Y=14.5)
    airfoil_stations = [
        (0.00, center),         # thick center body
        (0.10, center),         # still thick in center body
        (0.21, transition),     # blending to conventional airfoil
        (0.41, outboard),       # outboard root
        (0.69, mid_outboard),   # thinner outboard
        (1.00, tip),            # thin tip
    ]

    # ── Build wing from guide curves ─────────────────────────────
    bwb = MultiSegmentWing.from_planform_curves(
        le_points=le_points,
        te_points=te_points,
        airfoil_stations=airfoil_stations,
        num_sections=40,        # dense sections for smooth loft
        name="BWB (Guide Curves)",
    )

    # ── GVM Analysis ─────────────────────────────────────────────
    num_pts = 60
    X, Y, Z = bwb.to_vertex_grids(num_points_profile=num_pts)
    triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
    volume = VolumeCalculator.compute_solid_volume(triangles)

    density = 150.0   # composite structure [kg/m^3]
    mass = volume * density
    cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

    print(f"Configuration: {bwb.name}")
    print(f"  Sections: {len(bwb.get_section_frames())}")
    print(f"  Volume: {volume:.4f} m^3")
    print(f"  Mass:   {mass:.1f} kg")
    print(f"  CG:     ({cg[0]:.3f}, {cg[1]:.3f}, {cg[2]:.3f}) m")
    print(f"  Ixx={inertia[0]:.1f}, Iyy={inertia[1]:.1f}, "
          f"Izz={inertia[2]:.1f} kg*m^2")

    # ── Compare with segment-based BWB ───────────────────────────
    print("\n  Note: guide-curve construction produces a smoother planform")
    print("  than the segment-based approach (no LE/TE kinks at segment")
    print("  boundaries).")

    # ── NURBS export ─────────────────────────────────────────────
    os.makedirs(EXPORT_DIR, exist_ok=True)
    shape = bwb.to_occ_shape()
    step_path = os.path.join(EXPORT_DIR, "blended_wing_body_guided.step")
    NurbsExporter.to_step(shape, step_path)
    print(f"\n  STEP exported: {step_path}")

    # ── Visualize ────────────────────────────────────────────────
    show_interactive(triangles, volume, mass, cg, inertia,
                     title=bwb.name)


if __name__ == "__main__":
    main()