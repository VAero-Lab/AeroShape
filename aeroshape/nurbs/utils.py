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


def sample_shape_grid(shape, n_spanwise, n_chordwise,
                      spanwise_clustering=None, chordwise_clustering=None):
    """Sample an OCC shape on a structured (u, v) parametric grid.

    Extracts the lateral (non-end-cap) faces from a lofted shape, sorts
    them by their spanwise extent, and evaluates each face on a parametric
    grid.  Clustering distribution laws can be applied in both directions.

    When the loft contains multiple lateral faces (e.g. one per segment),
    the spanwise samples are distributed proportionally across them and
    the results are stitched into a single contiguous grid.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OCC shape (lofted shell or solid).
    n_spanwise : int
        Total number of spanwise stations (v direction).
    n_chordwise : int
        Number of chordwise points per station (u direction).
    spanwise_clustering : callable or None
        Distribution law ``f(n) -> np.ndarray`` from
        :mod:`aeroshape.analysis.clustering`.
    chordwise_clustering : callable or None
        Distribution law ``f(n) -> np.ndarray`` from
        :mod:`aeroshape.analysis.clustering`.

    Returns
    -------
    X, Y, Z : np.ndarray
        Coordinate matrices of shape (n_spanwise, n_chordwise).
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    # --- Collect all faces and identify lateral (skin) faces ---
    faces_info = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)

        u_min = adaptor.FirstUParameter()
        u_max = adaptor.LastUParameter()
        v_min = adaptor.FirstVParameter()
        v_max = adaptor.LastVParameter()

        # Evaluate midpoint to determine spanwise (Y) range
        pt_v0 = adaptor.Value(0.5 * (u_min + u_max), v_min)
        pt_v1 = adaptor.Value(0.5 * (u_min + u_max), v_max)
        y_span = abs(pt_v1.Y() - pt_v0.Y())

        # Compute face area for end-cap filtering
        gprops = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, gprops)
        area = gprops.Mass()

        faces_info.append({
            'face': face,
            'adaptor': adaptor,
            'u_range': (u_min, u_max),
            'v_range': (v_min, v_max),
            'y_start': min(pt_v0.Y(), pt_v1.Y()),
            'y_span': y_span,
            'area': area,
        })
        explorer.Next()

    if not faces_info:
        raise ValueError("Shape contains no faces")

    # Filter out end-cap faces: keep only faces with significant span
    max_span = max(fi['y_span'] for fi in faces_info)
    if max_span > 1e-6:
        lateral = [fi for fi in faces_info if fi['y_span'] > 0.1 * max_span]
    else:
        lateral = faces_info

    if not lateral:
        lateral = faces_info

    # Sort lateral faces by their spanwise start position
    lateral.sort(key=lambda fi: fi['y_start'])

    # --- Distribute spanwise samples across faces ---
    total_span = sum(fi['y_span'] for fi in lateral)

    X_rows = []
    Y_rows = []
    Z_rows = []

    # Compute how many spanwise samples go to each face
    if len(lateral) == 1:
        samples_per_face = [n_spanwise]
    else:
        samples_per_face = []
        assigned = 0
        for k, fi in enumerate(lateral):
            if k == len(lateral) - 1:
                n_face = n_spanwise - assigned
            else:
                frac = fi['y_span'] / total_span if total_span > 0 else 1.0
                n_face = max(2, round(frac * n_spanwise))
            samples_per_face.append(n_face)
            assigned += n_face

    for k, fi in enumerate(lateral):
        adaptor = fi['adaptor']
        u_min, u_max = fi['u_range']
        v_min, v_max = fi['v_range']
        n_face = samples_per_face[k]

        # Chordwise parameter values (u direction)
        if chordwise_clustering is not None:
            u_norm = chordwise_clustering(n_chordwise)
        else:
            u_norm = np.linspace(0.0, 1.0, n_chordwise)
        u_vals = u_min + u_norm * (u_max - u_min)

        # Spanwise parameter values (v direction) for this face
        if spanwise_clustering is not None:
            v_norm_full = spanwise_clustering(n_face)
        else:
            v_norm_full = np.linspace(0.0, 1.0, n_face)
        v_vals = v_min + v_norm_full * (v_max - v_min)

        # Skip the first station if this is not the first face
        # (it overlaps the last station of the previous face)
        start_idx = 1 if k > 0 else 0

        for j in range(start_idx, n_face):
            row_x = np.zeros(n_chordwise)
            row_y = np.zeros(n_chordwise)
            row_z = np.zeros(n_chordwise)
            for i in range(n_chordwise):
                pt = adaptor.Value(u_vals[i], v_vals[j])
                row_x[i] = pt.X()
                row_y[i] = pt.Y()
                row_z[i] = pt.Z()
            X_rows.append(row_x)
            Y_rows.append(row_y)
            Z_rows.append(row_z)

    X = np.array(X_rows)
    Y = np.array(Y_rows)
    Z = np.array(Z_rows)
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
