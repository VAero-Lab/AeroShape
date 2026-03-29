"""Strut-Braced Wing (SBW) / Truss-Braced Wing configuration.

Models a high-aspect-ratio wing with a supporting strut, similar to
configurations in Gagnon & Zingg (AIAA 2013-2850, Fig. 10) and
NASA/Boeing SUGAR / X-66A concepts.

The strut sweep and dihedral are computed so that the strut tip reaches
the wing underside at the strut attachment point (~60 % semi-span).

Uses AircraftModel to assemble:
- Main wing: high aspect ratio, moderate sweep
- Strut: thin section connecting fuselage belly to mid-span of main wing
"""

import math
import os

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
from aeroshape import MeshTopologyManager

EXPORT_DIR = "Exports"


def main():
    # ── Airfoil profiles ──────────────────────────────────────────
    main_root = AirfoilProfile.from_naca4("2412", num_points=50)
    main_mid = AirfoilProfile.from_naca4("2410", num_points=50)
    main_tip = AirfoilProfile.from_naca4("0008", num_points=50)
    strut_profile = AirfoilProfile.from_naca4("0006", num_points=50)

    # ── Main wing parameters ─────────────────────────────────────
    inboard_span = 7.0           # root to strut attachment [m]
    outboard_span = 10.0         # strut attachment to near-tip [m]
    wingtip_span = 0.6           # wingtip fairing [m]
    inboard_sweep_deg = 12.0
    inboard_dihedral_deg = 2.0
    inboard_root_chord = 3.0
    inboard_tip_chord = 2.2      # chord at strut attachment

    # ── Main wing (high aspect ratio) ────────────────────────────
    main_wing = MultiSegmentWing(name="Main Wing")

    # Inboard (root → strut attachment point)
    main_wing.add_segment(SegmentSpec(
        span=inboard_span,
        root_airfoil=main_root,
        tip_airfoil=main_mid,
        root_chord=inboard_root_chord,
        tip_chord=inboard_tip_chord,
        sweep_le_deg=inboard_sweep_deg,
        dihedral_deg=inboard_dihedral_deg,
        twist_deg=-1.0,
        num_sections=10,
    ))

    # Outboard (strut attachment → near-tip)
    main_wing.add_segment(SegmentSpec(
        span=outboard_span,
        root_airfoil=main_mid,
        tip_airfoil=main_tip,
        root_chord=inboard_tip_chord,
        tip_chord=0.8,
        sweep_le_deg=15.0,
        dihedral_deg=4.0,
        twist_deg=-2.0,
        num_sections=12,
    ))

    # Wingtip fairing
    main_wing.add_segment(SegmentSpec(
        span=wingtip_span,
        root_airfoil=main_tip,
        tip_airfoil=main_tip,
        root_chord=0.8,
        tip_chord=0.3,
        sweep_le_deg=40.0,
        dihedral_deg=60.0,
        num_sections=5,
    ))

    # ── Strut geometry (computed to meet wing at attachment) ──────
    # Wing position at strut attachment point (Y = inboard_span)
    wing_le_x = inboard_span * math.tan(math.radians(inboard_sweep_deg))
    wing_z = inboard_span * math.tan(math.radians(inboard_dihedral_deg))
    wing_qc_x = wing_le_x + inboard_tip_chord * 0.25   # quarter-chord

    # Strut root position (on fuselage belly, slightly aft of wing LE)
    strut_origin_x = 1.0    # aft of wing root LE [m]
    strut_origin_z = -2.0   # below wing root [m]

    # Strut must span from origin to wing underside at attachment
    strut_span = inboard_span   # same Y extent
    strut_sweep_deg = math.degrees(
        math.atan((wing_qc_x - strut_origin_x) / strut_span)
    )
    strut_dihedral_deg = math.degrees(
        math.atan((wing_z - strut_origin_z) / strut_span)
    )

    # Verify strut tip meets wing
    strut_tip_x = strut_origin_x + strut_span * math.tan(
        math.radians(strut_sweep_deg))
    strut_tip_z = strut_origin_z + strut_span * math.tan(
        math.radians(strut_dihedral_deg))
    print(f"Strut tip:  X={strut_tip_x:.3f}, Z={strut_tip_z:.3f}")
    print(f"Wing at Y=7: LE_X={wing_le_x:.3f}, QC_X={wing_qc_x:.3f}, "
          f"Z={wing_z:.3f}")
    print(f"Strut sweep={strut_sweep_deg:.1f} deg, "
          f"dihedral={strut_dihedral_deg:.1f} deg")

    strut = MultiSegmentWing(name="Strut")
    strut.add_segment(SegmentSpec(
        span=strut_span,
        root_airfoil=strut_profile,
        tip_airfoil=strut_profile,
        root_chord=0.8,
        tip_chord=0.5,
        sweep_le_deg=strut_sweep_deg,
        dihedral_deg=strut_dihedral_deg,
        twist_deg=0.0,
        num_sections=10,
    ))

    # ── Aircraft assembly ────────────────────────────────────────
    aircraft = AircraftModel(name="Strut-Braced Wing")
    aircraft.add_surface(main_wing, origin=(0.0, 0.0, 0.0))
    aircraft.add_surface(strut, origin=(strut_origin_x, 0.0, strut_origin_z))

    # ── GVM Analysis ─────────────────────────────────────────────
    num_pts = 50
    triangles = aircraft.to_triangles(num_points_profile=num_pts)
    volume = VolumeCalculator.compute_solid_volume(triangles)
    density = 2700.0
    mass = volume * density

    grids = aircraft.to_vertex_grids_list(num_points_profile=num_pts)

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
    os.makedirs(EXPORT_DIR, exist_ok=True)
    shape = aircraft.to_occ_shape(fuse=False)
    step_path = os.path.join(EXPORT_DIR, "strut_braced_wing.step")
    NurbsExporter.to_step(shape, step_path)
    print(f"\n  STEP exported: {step_path}")

    # ── Visualize ────────────────────────────────────────────────
    cg_all = (0, 0, 0)
    inertia_all = (0, 0, 0, 0, 0, 0)
    show_interactive(triangles, volume, mass, cg_all, inertia_all,
                     title=aircraft.name)


if __name__ == "__main__":
    main()
