"""Native NURBS export via pythonocc/OCP.

Exports OCC shapes to STEP, IGES, STL, and BREP formats, preserving
exact NURBS geometry in STEP/IGES (no tessellation round-trip).
"""

from pathlib import Path


class NurbsExporter:
    """Export OCC TopoDS_Shape objects to various CAD formats."""

    @staticmethod
    def to_step(shape, filepath, units="mm"):
        """Export to STEP format (AP214, preserves NURBS geometry).

        Parameters
        ----------
        shape : TopoDS_Shape
            The shape to export.
        filepath : str or Path
            Output file path.
        units : str
            Unit system: "mm" or "m".
        """
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static

        writer = STEPControl_Writer()

        if units.lower() == "mm":
            Interface_Static.SetCVal_s("write.step.unit", "MM")
        else:
            Interface_Static.SetCVal_s("write.step.unit", "M")

        writer.Transfer(shape, STEPControl_AsIs)
        status = writer.Write(str(filepath))
        if status != 1:  # IFSelect_RetDone
            raise RuntimeError(f"STEP export failed with status {status}")

    @staticmethod
    def to_iges(shape, filepath):
        """Export to IGES format (preserves NURBS geometry).

        Parameters
        ----------
        shape : TopoDS_Shape
            The shape to export.
        filepath : str or Path
            Output file path.
        """
        from OCP.IGESControl import IGESControl_Writer

        writer = IGESControl_Writer()
        writer.AddShape(shape)
        writer.ComputeModel()
        ok = writer.Write(str(filepath))
        if not ok:
            raise RuntimeError(f"IGES export failed for {filepath}")

    @staticmethod
    def to_stl(shape, filepath, linear_deflection=0.1, angular_deflection=0.5):
        """Export to STL format (tessellated).

        Parameters
        ----------
        shape : TopoDS_Shape
            The shape to export.
        filepath : str or Path
            Output file path.
        linear_deflection : float
            Tessellation accuracy (chord deviation).
        angular_deflection : float
            Tessellation angular accuracy in radians.
        """
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer

        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False,
                                        angular_deflection, True)
        mesh.Perform()

        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        writer.Write(shape, str(filepath))

    @staticmethod
    def to_brep(shape, filepath):
        """Export to BREP format (OpenCASCADE native).

        Parameters
        ----------
        shape : TopoDS_Shape
            The shape to export.
        filepath : str or Path
            Output file path.
        """
        from OCP.BRepTools import BRepTools

        BRepTools.Write_s(shape, str(filepath))
