"""Export a wing design to STL, IGES, and STEP formats.

This example generates a wing mesh and exports it in all supported
CAD formats, demonstrating the exporter module without requiring the GUI.

- STL: ASCII mesh format (always available, requires only numpy)
- IGES/STEP: NURBS-based CAD formats (require gmsh: pip install gmsh)

The exported files can be imported into any CAD software (Fusion 360,
SolidWorks, FreeCAD, etc.) for further analysis or manufacturing.
"""

from aeroshape import (
    WingMeshFactory,
    MeshTopologyManager,
    VolumeCalculator,
    ModelExporter,
)


def main():
    # Generate wing
    X, Y, Z = WingMeshFactory.create(
        naca_root="2412", naca_tip="0012",
        semi_span=8.0, chord_root=2.5, chord_tip=0.8,
        sweep_angle_deg=20.0, num_points_profile=50, num_sections=20
    )

    # Triangulate
    triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)

    # Compute properties
    volume = VolumeCalculator.compute_solid_volume(triangles)
    density = 2700.0  # aluminum
    mass = volume * density

    print(f"Wing volume: {volume:.6f} m^3")
    print(f"Wing mass (aluminum): {mass:.2f} kg")
    print(f"Triangle count: {len(triangles)}")

    vol_str = f"{volume:.6f}"
    mass_str = f"{mass:.3f}"

    # --- STL export (always available) ---
    stl_content = ModelExporter.export_to_stl(triangles, vol_str, mass_str)
    saved = ModelExporter.save_local_file("wing_export.stl", stl_content)
    print(f"\nSTL saved to: {saved}")

    # --- IGES and STEP export (require gmsh) ---
    try:
        iges_data = ModelExporter.export_to_iges(
            triangles, "wing", X, Y, Z, is_solid=True
        )
        saved = ModelExporter.save_local_file("wing_export.iges", iges_data)
        print(f"IGES saved to: {saved}")

        step_data = ModelExporter.export_to_step(
            triangles, "wing", X, Y, Z, is_solid=True
        )
        saved = ModelExporter.save_local_file("wing_export.step", step_data)
        print(f"STEP saved to: {saved}")
    except ImportError:
        print("\nIGES/STEP export requires gmsh. Install with: pip install gmsh")


if __name__ == "__main__":
    main()
