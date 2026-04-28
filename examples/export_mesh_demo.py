"""Export the structured visualization mesh to STL and CGNS formats.

Demonstrates the mesh export pipeline that writes the same triangulated
surface used for rendering — a structured chordwise × spanwise mesh
suitable for FEM/CFD surface meshing.

The STL export has zero extra dependencies.
The CGNS export requires ``pip install h5py``.
"""

import os

from aeroshape import (
    AirfoilProfile,
    SegmentSpec,
    MultiSegmentWing,
)

EXPORT_DIR = "Exports"


def main():
    # ── Define wing geometry ──────────────────────────────────────
    root = AirfoilProfile.from_naca4("2412", num_points=80)
    tip = AirfoilProfile.from_naca4("0012", num_points=80)

    wing = MultiSegmentWing(name="Demo_Wing")
    wing.add_segment(SegmentSpec(
        span=8.0,
        root_airfoil=root,
        tip_airfoil=tip,
        root_chord=2.5,
        tip_chord=0.8,
        sweep_le_deg=20.0,
        num_sections=20,
    ))

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # ── Export structured mesh to STL ─────────────────────────────
    stl_path = os.path.join(EXPORT_DIR, "wing_mesh.stl")
    wing.export_mesh_stl(stl_path, num_points_profile=80, closed=True)

    # ── Export structured mesh to CGNS ────────────────────────────
    cgns_path = os.path.join(EXPORT_DIR, "wing_mesh.cgns")
    wing.export_mesh_cgns(cgns_path, num_points_profile=80, closed=False)

    # ── Export an open shell (no end-caps) ────────────────────────
    stl_open = os.path.join(EXPORT_DIR, "wing_mesh_open.stl")
    wing.export_mesh_stl(stl_open, num_points_profile=80, closed=False)
    print(f"Open shell STL: {stl_open}")

    # ── Print mesh statistics ─────────────────────────────────────
    triangles = wing.to_triangles(num_points_profile=80, closed=True)
    X, Y, Z = wing.to_vertex_grids(num_points_profile=80)
    n_verts = X.shape[0] * X.shape[1] + 2  # +2 for end-cap centers
    print(f"\nMesh statistics:")
    print(f"  Grid shape:     {X.shape[0]} sections × {X.shape[1]} points")
    print(f"  Vertices:       {n_verts}")
    print(f"  Triangles:      {len(triangles)}")


if __name__ == "__main__":
    main()
