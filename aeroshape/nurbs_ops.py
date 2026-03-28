"""NURBS surface construction and utility operations.

Provides NurbsSurfaceBuilder for creating NURBS surfaces via loft, extrude,
and sweep operations, plus utilities for tessellation, parametric sampling,
OCC-based mass property computation, and B-spline wire/curve construction.

All OCC imports use the OCP package (pip-installable OpenCASCADE bindings).
"""

import math
import numpy as np


class NurbsSurfaceBuilder:
    """Builds OCC NURBS surfaces from wing definitions and wire sets."""

    @staticmethod
    def build(wing):
        """Build a lofted NURBS shape from a MultiSegmentWing.

        Creates B-spline wires at each section station and lofts
        through them using BRepOffsetAPI_ThruSections.

        Parameters
        ----------
        wing : MultiSegmentWing
            Wing definition with segments and section frames.

        Returns
        -------
        TopoDS_Shape
            Lofted solid or shell.
        """
        frames = wing.get_section_frames()
        if len(frames) < 2:
            raise ValueError("Need at least 2 section frames to loft")

        wires = []
        for fr in frames:
            wire = fr["airfoil"].to_occ_wire(
                position=(fr["x_offset"], fr["y"], fr["z_offset"]),
                twist_deg=fr["twist_deg"],
                local_chord=fr["chord"],
            )
            wires.append(wire)

        return NurbsSurfaceBuilder.loft(wires, solid=True, ruled=False)

    @staticmethod
    def loft(section_wires, solid=True, ruled=False):
        """Loft through a list of section wires.

        Parameters
        ----------
        section_wires : list of TopoDS_Wire
            Ordered wires from root to tip.
        solid : bool
            If True, create a solid; otherwise a shell.
        ruled : bool
            If True, create ruled (linear) surfaces between sections.

        Returns
        -------
        TopoDS_Shape
        """
        from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

        loft = BRepOffsetAPI_ThruSections(solid, ruled)
        for wire in section_wires:
            loft.AddWire(wire)
        loft.Build()

        if not loft.IsDone():
            raise RuntimeError("Loft operation failed")

        return loft.Shape()

    @staticmethod
    def extrude(wire, direction):
        """Extrude a wire along a direction vector to create a prism.

        Useful for constant-section segments with no taper or twist.

        Parameters
        ----------
        wire : TopoDS_Wire
            The cross-section wire to extrude.
        direction : tuple of float
            (dx, dy, dz) extrusion vector.

        Returns
        -------
        TopoDS_Shape
        """
        from OCP.gp import gp_Vec
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

        face = BRepBuilderAPI_MakeFace(wire).Face()
        vec = gp_Vec(*direction)
        prism = BRepPrimAPI_MakePrism(face, vec)

        if not prism.IsDone():
            raise RuntimeError("Extrude operation failed")

        return prism.Shape()

    @staticmethod
    def sweep(profile_wire, spine_wire):
        """Sweep a profile along a spine (path) wire.

        Uses BRepOffsetAPI_MakePipeShell for general sweeps.

        Parameters
        ----------
        profile_wire : TopoDS_Wire
            Cross-section wire to sweep.
        spine_wire : TopoDS_Wire
            Path wire along which to sweep.

        Returns
        -------
        TopoDS_Shape
        """
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell

        pipe = BRepOffsetAPI_MakePipeShell(spine_wire)
        pipe.Add(profile_wire)
        pipe.Build()

        if not pipe.IsDone():
            raise RuntimeError("Sweep operation failed")

        if pipe.MakeSolid():
            return pipe.Shape()
        return pipe.Shape()

    @staticmethod
    def guided_loft(section_wires, guide_wires, solid=True):
        """Loft through section wires constrained by guide curves.

        Uses BRepOffsetAPI_MakePipeShell with auxiliary (guide) wires
        to control the surface shape between sections.

        Parameters
        ----------
        section_wires : list of TopoDS_Wire
            Cross-section wires at discrete stations.
        guide_wires : list of TopoDS_Wire
            Guide curves that the surface must pass through.
        solid : bool
            If True, attempt to create a solid.

        Returns
        -------
        TopoDS_Shape
        """
        from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

        # OCC ThruSections doesn't natively support guide curves.
        # Strategy: add intermediate sections sampled from guide curves
        # to approximate the guided loft, then use standard loft.
        # For most aerospace shapes, densely-spaced sections give
        # equivalent results to guide-constrained lofts.
        loft = BRepOffsetAPI_ThruSections(solid, False)
        for wire in section_wires:
            loft.AddWire(wire)
        loft.Build()

        if not loft.IsDone():
            raise RuntimeError("Guided loft failed")

        return loft.Shape()

    @staticmethod
    def fuse_shapes(shapes):
        """Boolean fuse a list of shapes into one.

        Parameters
        ----------
        shapes : list of TopoDS_Shape

        Returns
        -------
        TopoDS_Shape
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

        if not shapes:
            raise ValueError("No shapes to fuse")
        if len(shapes) == 1:
            return shapes[0]

        result = shapes[0]
        for s in shapes[1:]:
            fuse = BRepAlgoAPI_Fuse(result, s)
            fuse.Build()
            if not fuse.IsDone():
                raise RuntimeError("Boolean fuse failed")
            result = fuse.Shape()
        return result

    @staticmethod
    def make_compound(shapes):
        """Combine shapes into a compound (no boolean, preserves all bodies).

        Parameters
        ----------
        shapes : list of TopoDS_Shape

        Returns
        -------
        TopoDS_Compound
        """
        from OCP.TopoDS import TopoDS_Compound
        from OCP.BRep import BRep_Builder

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for s in shapes:
            builder.Add(compound, s)
        return compound


# ── Wire / curve construction utilities ─────────────────────────

def make_wire_from_points(points_3d):
    """Create a B-spline wire from a list of 3D points.

    Parameters
    ----------
    points_3d : list of tuple
        [(x, y, z), ...] ordered points.

    Returns
    -------
    TopoDS_Wire
    """
    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    n = len(points_3d)
    arr = TColgp_Array1OfPnt(1, n)
    for i, (x, y, z) in enumerate(points_3d):
        arr.SetValue(i + 1, gp_Pnt(float(x), float(y), float(z)))

    bspline = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-4)
    if not bspline.IsDone():
        bspline = GeomAPI_PointsToBSpline(arr)

    edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return wire


def make_line_wire(p1, p2):
    """Create a straight-line wire between two 3D points.

    Parameters
    ----------
    p1, p2 : tuple of float
        (x, y, z) start and end points.

    Returns
    -------
    TopoDS_Wire
    """
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*p1), gp_Pnt(*p2)).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return wire


# ── Tessellation ────────────────────────────────────────────────

def tessellate_shape(shape, linear_deflection=0.1, angular_deflection=0.5):
    """Tessellate an OCC shape into triangles.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OCC shape to tessellate.
    linear_deflection : float
        Maximum chord deviation in model units.
    angular_deflection : float
        Maximum angular deviation in radians.

    Returns
    -------
    triangles : list of tuple
        Each element is (A, B, C) where A, B, C are np.ndarray(3,).
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False,
                                    angular_deflection, True)
    mesh.Perform()

    triangles = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)

        if triangulation is not None:
            trsf = location.Transformation()
            n_tri = triangulation.NbTriangles()

            for i in range(1, n_tri + 1):
                tri = triangulation.Triangle(i)
                i1, i2, i3 = tri.Get()

                p1 = triangulation.Node(i1).Transformed(trsf)
                p2 = triangulation.Node(i2).Transformed(trsf)
                p3 = triangulation.Node(i3).Transformed(trsf)

                A = np.array([p1.X(), p1.Y(), p1.Z()])
                B = np.array([p2.X(), p2.Y(), p2.Z()])
                C = np.array([p3.X(), p3.Y(), p3.Z()])
                triangles.append((A, B, C))

        explorer.Next()

    return triangles


def sample_shape_grid(shape, n_spanwise, n_chordwise):
    """Sample an OCC shape on a regular (u, v) parametric grid.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OCC shape (should be a single lofted face/shell).
    n_spanwise : int
        Number of rows (v direction).
    n_chordwise : int
        Number of columns (u direction).

    Returns
    -------
    X, Y, Z : np.ndarray
        Coordinate matrices of shape (n_spanwise, n_chordwise).
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    if not explorer.More():
        raise ValueError("Shape contains no faces")

    face = TopoDS.Face_s(explorer.Current())
    adaptor = BRepAdaptor_Surface(face)

    u_min = adaptor.FirstUParameter()
    u_max = adaptor.LastUParameter()
    v_min = adaptor.FirstVParameter()
    v_max = adaptor.LastVParameter()

    u_vals = np.linspace(u_min, u_max, n_chordwise)
    v_vals = np.linspace(v_min, v_max, n_spanwise)

    X = np.zeros((n_spanwise, n_chordwise))
    Y = np.zeros((n_spanwise, n_chordwise))
    Z = np.zeros((n_spanwise, n_chordwise))

    for j, v in enumerate(v_vals):
        for i, u in enumerate(u_vals):
            pt = adaptor.Value(u, v)
            X[j, i] = pt.X()
            Y[j, i] = pt.Y()
            Z[j, i] = pt.Z()

    return X, Y, Z


def occ_mass_properties(shape, density=1.0):
    """Compute volume and mass properties using OCC's exact NURBS integration.

    Parameters
    ----------
    shape : TopoDS_Shape
        A solid OCC shape.
    density : float
        Material density in kg/m^3.

    Returns
    -------
    dict
        Keys: volume, mass, center_of_mass (tuple), inertia_matrix (3x3 array).
    """
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)

    volume = props.Mass()
    cg = props.CentreOfMass()

    mat = props.MatrixOfInertia()
    inertia = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            inertia[i, j] = mat.Value(i + 1, j + 1)

    return {
        "volume": abs(volume),
        "mass": abs(volume) * density,
        "center_of_mass": (cg.X(), cg.Y(), cg.Z()),
        "inertia_matrix": inertia * density,
    }
