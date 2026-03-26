"""3D model export and visualization utilities.

This module handles exporting triangulated lifting surface meshes to
standard CAD file formats (STL, IGES, STEP) and preparing triangle
data for interactive 3D visualization.

The STL export uses the ASCII format with triangle normals computed
from the vertex data. IGES and STEP exports use the gmsh library to
create true NURBS lofted surfaces from the wing section profiles.

Reference:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378.
"""

import os
import numpy as np


class ModelExporter:
    """Exports triangulated meshes to CAD file formats.

    Supports STL (ASCII), IGES, and STEP formats. The IGES/STEP
    exports produce NURBS-based geometry via gmsh, while STL is
    purely mesh-based.
    """

    @staticmethod
    def save_local_file(filename, data):
        """Save export data to the Exports/ directory with auto-numbering.

        If a file with the given name already exists, appends an
        incrementing counter (e.g. Wing_1.stl, Wing_2.stl).

        Parameters
        ----------
        filename : str
            Desired filename with extension (e.g. 'Wing.stl').
        data : bytes or str
            File content to write.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exports_dir = os.path.join(base_dir, "Exports")
        os.makedirs(exports_dir, exist_ok=True)

        name, ext = os.path.splitext(filename)
        file_path = os.path.join(exports_dir, filename)
        counter = 1

        while os.path.exists(file_path):
            file_path = os.path.join(exports_dir, f"{name}_{counter}{ext}")
            counter += 1

        with open(file_path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode('utf-8'))

        return file_path

    @staticmethod
    def export_to_stl(triangles, volume_str, mass_str):
        """Export a triangulated mesh to ASCII STL format.

        Embeds volume and mass metadata in the solid name for
        traceability when importing into CAD tools.

        Parameters
        ----------
        triangles : list of tuple
            List of (A, B, C) triangular facets.
        volume_str : str
            Computed volume as a formatted string.
        mass_str : str
            Computed mass as a formatted string.

        Returns
        -------
        str
            STL file content as a string.
        """
        solid_name = f"Wing_Vol_{volume_str}m3_Mass_{mass_str}kg"
        lines = [f"solid {solid_name}"]

        for A, B, C in triangles:
            n = np.cross(B - A, C - A)
            norm = np.linalg.norm(n)
            n = (n / norm) if norm > 1e-10 else np.array([0.0, 0.0, 0.0])
            lines.append(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            lines.append("    outer loop")
            lines.append(f"      vertex {A[0]:.6f} {A[1]:.6f} {A[2]:.6f}")
            lines.append(f"      vertex {B[0]:.6f} {B[1]:.6f} {B[2]:.6f}")
            lines.append(f"      vertex {C[0]:.6f} {C[1]:.6f} {C[2]:.6f}")
            lines.append("    endloop")
            lines.append("  endfacet")

        lines.append(f"endsolid {solid_name}")
        return "\n".join(lines)

    @staticmethod
    def _export_cad_format(triangles, fmt, X=None, Y=None, Z=None,
                           is_solid=True):
        """Export to IGES or STEP via gmsh NURBS lofting.

        When vertex matrices (X, Y, Z) are provided, creates true NURBS
        surfaces by lofting spline profiles through sections. Otherwise,
        falls back to individual triangle surfaces.

        The model is scaled x1000 (meters to millimeters) for compatibility
        with CAD tools that assume STEP/IGES units are in millimeters.

        Parameters
        ----------
        triangles : list of tuple
            Triangulated mesh (used as fallback if X, Y, Z are None).
        fmt : str
            Output format: 'iges' or 'step'.
        X, Y, Z : np.ndarray, optional
            Vertex matrices for NURBS lofting.
        is_solid : bool
            If True, create a solid body; otherwise, create surfaces only.

        Returns
        -------
        bytes
            File content as bytes.
        """
        import gmsh
        import tempfile

        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add("wing_model")

            if X is not None and Y is not None and Z is not None:
                # NURBS loft through section profiles
                wires = []
                num_sections = X.shape[0]
                num_points = X.shape[1]

                for j in range(num_sections):
                    points = []
                    for i in range(num_points):
                        p = gmsh.model.occ.addPoint(
                            X[j, i], Y[j, i], Z[j, i]
                        )
                        points.append(p)
                    points.append(points[0])  # Close the profile
                    spline = gmsh.model.occ.addSpline(points)
                    wire = gmsh.model.occ.addWire([spline])
                    wires.append(wire)

                gmsh.model.occ.addThruSections(
                    wires, makeSolid=is_solid, makeRuled=False
                )
            else:
                # Fallback: individual triangle surfaces
                for t in triangles:
                    p1 = gmsh.model.occ.addPoint(t[0][0], t[0][1], t[0][2])
                    p2 = gmsh.model.occ.addPoint(t[1][0], t[1][1], t[1][2])
                    p3 = gmsh.model.occ.addPoint(t[2][0], t[2][1], t[2][2])

                    l1 = gmsh.model.occ.addLine(p1, p2)
                    l2 = gmsh.model.occ.addLine(p2, p3)
                    l3 = gmsh.model.occ.addLine(p3, p1)

                    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                    gmsh.model.occ.addPlaneSurface([cl])

            # Scale x1000: GVM uses meters, CAD tools expect millimeters
            ents = gmsh.model.occ.getEntities()
            gmsh.model.occ.dilate(ents, 0, 0, 0, 1000.0, 1000.0, 1000.0)

            gmsh.model.occ.synchronize()

            with tempfile.NamedTemporaryFile(
                suffix=f".{fmt}", delete=False
            ) as tmp:
                tmp_name = tmp.name

            gmsh.write(tmp_name)

            with open(tmp_name, 'rb') as f:
                data = f.read()

            os.remove(tmp_name)
            return data
        finally:
            gmsh.clear()
            gmsh.finalize()

    @staticmethod
    def export_to_iges(triangles, name, X=None, Y=None, Z=None,
                       is_solid=True):
        """Export to IGES format via gmsh.

        Parameters
        ----------
        triangles : list of tuple
            Triangulated mesh.
        name : str
            Model name (for metadata).
        X, Y, Z : np.ndarray, optional
            Vertex matrices for NURBS lofting.
        is_solid : bool
            Create solid body if True.

        Returns
        -------
        bytes
            IGES file content.
        """
        return ModelExporter._export_cad_format(
            triangles, 'iges', X, Y, Z, is_solid
        )

    @staticmethod
    def export_to_step(triangles, name, X=None, Y=None, Z=None,
                       is_solid=True):
        """Export to STEP format via gmsh.

        Parameters
        ----------
        triangles : list of tuple
            Triangulated mesh.
        name : str
            Model name (for metadata).
        X, Y, Z : np.ndarray, optional
            Vertex matrices for NURBS lofting.
        is_solid : bool
            Create solid body if True.

        Returns
        -------
        bytes
            STEP file content.
        """
        return ModelExporter._export_cad_format(
            triangles, 'step', X, Y, Z, is_solid
        )

    @staticmethod
    def prepare_plot_data(triangles):
        """Prepare triangle data for Plotly Mesh3d visualization.

        Converts a list of triangle tuples into flat arrays suitable
        for plotly.graph_objects.Mesh3d.

        Parameters
        ----------
        triangles : list of tuple
            List of (A, B, C) triangular facets.

        Returns
        -------
        x_pts, y_pts, z_pts : list of float
            Flattened vertex coordinates.
        i_idx, j_idx, k_idx : list of int
            Triangle connectivity indices.
        """
        x_pts, y_pts, z_pts = [], [], []
        i_idx, j_idx, k_idx = [], [], []

        for t_id, (A, B, C) in enumerate(triangles):
            x_pts.extend([A[0], B[0], C[0]])
            y_pts.extend([A[1], B[1], C[1]])
            z_pts.extend([A[2], B[2], C[2]])

            i_idx.append(t_id * 3)
            j_idx.append(t_id * 3 + 1)
            k_idx.append(t_id * 3 + 2)

        return x_pts, y_pts, z_pts, i_idx, j_idx, k_idx
