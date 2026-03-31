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
        """Export to STEP format (AP242, preserves NURBS geometry)."""
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
        from OCP.STEPCAFControl import STEPCAFControl_Writer
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.Interface import Interface_Static

        # Optimization flags
        Interface_Static.SetIVal_s("write.step.schema", 3)   # AP242
        Interface_Static.SetIVal_s("write.precision.mode", 0)
        Interface_Static.SetIVal_s("write.step.assembly", 1)

        occ_shape = _unwrap_shape(shape)
        
        # Create XDE Document for efficient assembly handling
        app = XCAFApp_Application.GetApplication_s()
        doc = TDocStd_Document(TCollection_ExtendedString("XCAF"))
        app.InitDocument(doc)
        
        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        # The 'False' here means don't expand compounds into individual parts unless necessary
        # but we want the assembly structure if possible.
        shape_tool.SetShape(shape_tool.NewShape(), occ_shape)
        
        writer = STEPCAFControl_Writer()
        if units.lower() == "mm":
            Interface_Static.SetCVal_s("write.step.unit", "MM")
        else:
            Interface_Static.SetCVal_s("write.step.unit", "M")

        writer.Transfer(doc)
        status = writer.Write(str(filepath))
        if status != 1:  # IFSelect_RetDone
            raise RuntimeError(f"STEP export failed with status {status}")

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
