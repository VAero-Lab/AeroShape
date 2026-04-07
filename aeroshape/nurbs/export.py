"""Native NURBS export via pythonocc/OCP.

Exports OCC shapes to STEP, IGES, STL, and BREP formats, preserving
exact NURBS geometry in STEP/IGES (no tessellation round-trip).
"""

from pathlib import Path


def _unwrap_shape(shape):
    """Access the underlying TopoDS_Shape if it's a build123d/OCP wrapper."""
    return getattr(shape, "wrapped", shape)


class NurbsExporter:
    """Export OCC TopoDS_Shape objects to various CAD formats."""

    @staticmethod
    def to_step(shape, filepath, units="mm"):
        """Export to STEP format (AP242, preserves NURBS geometry).

        Uses STEPControl_Writer (direct transfer) which works reliably
        for all shape types — solids, shells, compounds, and assemblies.
        Suppresses verbose OCC transfer statistics for clean CLI output.
        """
        import os
        import sys
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static

        # 1. Capture and redirect C-level stdout to suppress transfer stats
        fd = sys.stdout.fileno()
        old_stdout = os.dup(fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fd)
        os.close(devnull)

        try:
            # 2. Export configuration
            Interface_Static.SetIVal_s("write.step.schema", 5)  # AP242
            Interface_Static.SetIVal_s("write.precision.mode", 0)
            Interface_Static.SetIVal_s("write.step.assembly", 1)

            if units.lower() == "mm":
                Interface_Static.SetCVal_s("write.step.unit", "MM")
            else:
                Interface_Static.SetCVal_s("write.step.unit", "M")

            occ_shape = _unwrap_shape(shape)

            # 3. Direct transfer — works for solids, shells, and compounds
            writer = STEPControl_Writer()
            writer.Transfer(occ_shape, STEPControl_AsIs)
            status = writer.Write(str(filepath))

            if status != 1:  # IFSelect_RetDone
                raise RuntimeError(f"STEP export failed with status {status}")

        finally:
            # 4. Restore stdout
            try:
                os.dup2(old_stdout, fd)
            except OSError:
                pass
            try:
                os.close(old_stdout)
            except OSError:
                pass

    @staticmethod
    def to_iges(shape, filepath):
        """Export to IGES format (preserves NURBS geometry)."""
        from OCP.IGESControl import IGESControl_Writer

        writer = IGESControl_Writer()
        writer.AddShape(_unwrap_shape(shape))
        writer.ComputeModel()
        ok = writer.Write(str(filepath))
        if not ok:
            raise RuntimeError(f"IGES export failed for {filepath}")

    @staticmethod
    def to_stl(shape, filepath, linear_deflection=0.1, angular_deflection=0.5):
        """Export to STL format (tessellated)."""
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer

        occ_shape = _unwrap_shape(shape)
        mesh = BRepMesh_IncrementalMesh(occ_shape, linear_deflection, False,
                                        angular_deflection, True)
        mesh.Perform()

        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        writer.Write(occ_shape, str(filepath))

    @staticmethod
    def to_brep(shape, filepath):
        """Export to BREP format (OpenCASCADE native)."""
        from OCP.BRepTools import BRepTools

        BRepTools.Write_s(_unwrap_shape(shape), str(filepath))
