"""NURBS utility functions: tessellation, sampling, mass properties, wire construction."""

import numpy as np


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
