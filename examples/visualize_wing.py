"""Wing visualization examples — interactive and static.

Demonstrates both visualization backends:
  1. Interactive 3D viewer (vedo/VTK) — CAD-like rotate, zoom, pan.
  2. Static matplotlib figure — publication-quality for papers and reports.

Usage:
    python examples/visualize_wing.py              # interactive (default)
    python examples/visualize_wing.py --static     # matplotlib figure
    python examples/visualize_wing.py --both       # show both sequentially
    python examples/visualize_wing.py --save       # save outputs to Exports/
"""

import argparse

from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
    VolumeCalculator,
    MassPropertiesCalculator,
    show_interactive,
    show_static,
)
from aeroshape import MeshTopologyManager


def main():
    parser = argparse.ArgumentParser(description="AeroShape wing visualization")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--static", action="store_true", help="Matplotlib figure only")
    group.add_argument("--both", action="store_true", help="Show both viewers")
    parser.add_argument("--save", action="store_true", help="Save outputs to Exports/")
    args = parser.parse_args()

    # ── Wing definition ────────────────────────────────────────────
    profile = AirfoilProfile.from_naca4("2412", num_points=60)

    wing = MultiSegmentWing(name="NACA 2412 Wing")
    wing.add_segment(SegmentSpec(
        span=10.0,
        root_airfoil=profile,
        tip_airfoil=profile,
        root_chord=2.0,
        tip_chord=1.0,
        sweep_le_deg=15.0,
        num_sections=15,
    ))

    X, Y, Z = wing.to_vertex_grids(num_points_profile=60)
    triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
    volume = VolumeCalculator.compute_solid_volume(triangles)

    density = 2700.0
    mass = volume * density
    cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

    print(f"Volume : {volume:.6f} m3")
    print(f"Mass   : {mass:.2f} kg")
    print(f"CG     : ({cg[0]:.4f}, {cg[1]:.4f}, {cg[2]:.4f})")
    print()

    save_dir = "Exports" if args.save else None

    show_mpl = args.static or args.both
    show_vtk = (not args.static) or args.both

    if show_mpl:
        save_fig = f"{save_dir}/wing_figure.png" if save_dir else None
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        show_static(
            triangles, volume, mass, cg, inertia,
            title="NACA 2412 — Solid Wing Properties",
            save_path=save_fig,
        )

    if show_vtk:
        save_png = f"{save_dir}/wing_interactive.png" if save_dir else None
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        show_interactive(
            triangles, volume, mass, cg, inertia,
            title="AeroShape NACA 2412 Solid Wing",
            save_screenshot=save_png,
        )


if __name__ == "__main__":
    main()
