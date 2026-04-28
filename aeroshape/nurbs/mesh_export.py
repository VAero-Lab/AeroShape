"""Structured mesh export to STL and CGNS formats.

Exports the structured triangulated mesh produced by MeshTopologyManager
(the same discretization used for visualization) to industry-standard
surface mesh formats suitable for FEM/CFD pre-processing.

Two export paths:

- **STL** (binary): Universal surface mesh format.  Zero extra dependencies
  beyond NumPy.
- **CGNS** (HDF5): CFD General Notation System — the standard interchange
  format for CFD solvers (Pointwise, ANSYS, OpenFOAM, SU2, etc.).
  Requires ``h5py``.

Both writers operate on the discrete vertex grids / triangle lists and are
completely independent of the OCC-based ``NurbsExporter``.
"""

import struct
import numpy as np


class MeshExporter:
    """Export structured surface meshes to STL and CGNS."""

    # ── Binary STL ────────────────────────────────────────────────

    @staticmethod
    def to_stl(triangles, filepath, name="AeroShape"):
        """Write a binary STL file from a triangle list.

        Parameters
        ----------
        triangles : list of tuple
            Each element is ``(A, B, C)`` where A, B, C are array-like
            of shape (3,) — the same format returned by
            ``MeshTopologyManager.get_wing_triangles()``.
        filepath : str or pathlib.Path
            Output file path.
        name : str
            Model name embedded in the 80-byte STL header.
        """
        filepath = str(filepath)
        n_tri = len(triangles)

        with open(filepath, "wb") as f:
            # 80-byte header (padded with spaces)
            header = name.encode("ascii", errors="replace")[:80]
            header = header.ljust(80, b"\x00")
            f.write(header)

            # Triangle count (uint32)
            f.write(struct.pack("<I", n_tri))

            # Each facet: normal (3×float32) + 3 vertices (9×float32) + attr (uint16)
            for A, B, C in triangles:
                A = np.asarray(A, dtype=np.float32)
                B = np.asarray(B, dtype=np.float32)
                C = np.asarray(C, dtype=np.float32)

                # Outward normal via cross product
                normal = np.cross(B - A, C - A)
                norm_mag = np.linalg.norm(normal)
                if norm_mag > 1e-12:
                    normal /= norm_mag
                else:
                    normal = np.zeros(3, dtype=np.float32)

                f.write(struct.pack("<3f", *normal))
                f.write(struct.pack("<3f", *A))
                f.write(struct.pack("<3f", *B))
                f.write(struct.pack("<3f", *C))
                f.write(struct.pack("<H", 0))  # attribute byte count

        print(f"STL mesh exported: {filepath}  ({n_tri} triangles)")

    # ── CGNS / HDF5 ──────────────────────────────────────────────

    @staticmethod
    def to_cgns(grids, filepath, base_name="AeroShape"):
        """Write a CGNS/HDF5 file from structured vertex grids.

        Each grid becomes a separate unstructured Zone containing TRI_3
        surface elements.  The triangulation uses the same diagonal-split
        convention as ``MeshTopologyManager.get_wing_triangles()``.

        Uses the official pyCGNS library (CGNS.PAT + CGNS.MAP) to produce
        fully SIDS-compliant files readable by ParaView, Gmsh, Pointwise,
        ANSYS, SU2, and all other CGNS-compatible tools.

        Parameters
        ----------
        grids : list of tuple
            Each element is ``(X, Y, Z, zone_name, closed)`` where
            X, Y, Z are ``np.ndarray`` of shape ``(n_sections, m_points)``
            and ``zone_name`` is a string label for the CGNS zone.
            ``closed`` is a bool indicating whether to include end-cap
            triangles for a watertight mesh.
        filepath : str or pathlib.Path
            Output ``.cgns`` file path.
        base_name : str
            Name of the CGNS Base node.
        """
        try:
            import CGNS.PAT.cgnslib as CGL
            import CGNS.PAT.cgnskeywords as CGK
            import CGNS.MAP as MAP
        except ImportError:
            raise ImportError(
                "CGNS export requires 'pyCGNS'.  "
                "Install with:  pip install pyCGNS"
            )

        filepath = str(filepath)

        # ── Build SIDS-compliant tree ─────────────────────────
        tree = CGL.newCGNSTree()
        # CellDimension=2 (surface elements), PhysicalDimension=3
        base = CGL.newBase(tree, base_name, 2, 3)

        for grid_entry in grids:
            X, Y, Z, zone_name, closed = grid_entry
            _write_cgns_zone(base, X, Y, Z, zone_name, closed)

        # ── Serialize to HDF5 ────────────────────────────────
        MAP.save(filepath, tree)
        print(f"CGNS mesh exported: {filepath}  ({len(grids)} zone(s))")


def _build_tri_connectivity(X, Y, Z, closed=True):
    """Build deduplicated vertex array and triangle connectivity.

    Uses the same algorithm as ``MeshTopologyManager.get_wing_triangles()``
    but returns vertex coordinates and 0-based face indices instead of
    coordinate tuples.

    Parameters
    ----------
    X, Y, Z : np.ndarray, shape (n_sections, m_points)
        Structured vertex grids.
    closed : bool
        If True, add end-cap fan triangles at root and tip.

    Returns
    -------
    coords : np.ndarray, shape (n_verts, 3)
        Unique vertex coordinates.
    faces : np.ndarray, shape (n_tris, 3)
        0-based vertex indices for each triangle.
    """
    n_sec, m_pts = X.shape

    # ── 1. Collect grid vertices (row-major: vertex[j, i] = j * m_pts + i) ──
    coords_list = []
    for j in range(n_sec):
        for i in range(m_pts):
            coords_list.append([X[j, i], Y[j, i], Z[j, i]])

    def _idx(j, i):
        return j * m_pts + i

    faces = []

    # ── 2. Outer surface quads → 2 triangles each ───────────────
    for j in range(n_sec - 1):
        for i in range(m_pts - 1):
            a = _idx(j, i)
            b = _idx(j, i + 1)
            c = _idx(j + 1, i)
            d = _idx(j + 1, i + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])

    # ── 3. Trailing-edge closure strips ─────────────────────────
    for j in range(n_sec - 1):
        a = _idx(j, 0)
        b = _idx(j, m_pts - 1)
        c = _idx(j + 1, 0)
        d = _idx(j + 1, m_pts - 1)

        A = np.array(coords_list[a])
        B = np.array(coords_list[b])
        C = np.array(coords_list[c])
        D = np.array(coords_list[d])

        if np.linalg.norm(A - B) > 1e-6:
            n1 = np.cross(B - A, C - A)
            if n1[0] < 0:
                faces.append([a, c, b])
            else:
                faces.append([a, b, c])
            n2 = np.cross(D - B, C - B)
            if n2[0] < 0:
                faces.append([b, c, d])
            else:
                faces.append([b, d, c])

    # ── 4. End-cap fan triangulations ───────────────────────────
    if closed:
        # Root cap (j=0)
        center_root = np.array([
            np.mean(X[0, :]), np.mean(Y[0, :]), np.mean(Z[0, :])
        ])
        root_center_idx = len(coords_list)
        coords_list.append(center_root.tolist())

        for i in range(m_pts - 1):
            a = _idx(0, i)
            b = _idx(0, i + 1)
            A = np.array(coords_list[a])
            B = np.array(coords_list[b])
            ntest = np.cross(B - center_root, A - center_root)
            if ntest[1] < 0:
                faces.append([root_center_idx, b, a])
            else:
                faces.append([root_center_idx, a, b])

        # Close root trailing-edge gap
        a = _idx(0, m_pts - 1)
        b = _idx(0, 0)
        A = np.array(coords_list[a])
        B = np.array(coords_list[b])
        if np.linalg.norm(A - B) > 1e-6:
            ntest = np.cross(B - center_root, A - center_root)
            if ntest[1] < 0:
                faces.append([root_center_idx, b, a])
            else:
                faces.append([root_center_idx, a, b])

        # Tip cap (j = n_sec - 1)
        j_tip = n_sec - 1
        center_tip = np.array([
            np.mean(X[j_tip, :]), np.mean(Y[j_tip, :]), np.mean(Z[j_tip, :])
        ])
        tip_center_idx = len(coords_list)
        coords_list.append(center_tip.tolist())

        for i in range(m_pts - 1):
            a = _idx(j_tip, i)
            b = _idx(j_tip, i + 1)
            A = np.array(coords_list[a])
            B = np.array(coords_list[b])
            ntest = np.cross(B - center_tip, A - center_tip)
            if ntest[1] > 0:
                faces.append([tip_center_idx, b, a])
            else:
                faces.append([tip_center_idx, a, b])

        # Close tip trailing-edge gap
        a = _idx(j_tip, m_pts - 1)
        b = _idx(j_tip, 0)
        A = np.array(coords_list[a])
        B = np.array(coords_list[b])
        if np.linalg.norm(A - B) > 1e-6:
            ntest = np.cross(B - center_tip, A - center_tip)
            if ntest[1] > 0:
                faces.append([tip_center_idx, b, a])
            else:
                faces.append([tip_center_idx, a, b])

    coords = np.array(coords_list, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)
    return coords, faces


def _write_cgns_zone(base_node, X, Y, Z, zone_name, closed=True):
    """Write a single unstructured zone with TRI_3 elements using pyCGNS.

    Parameters
    ----------
    base_node : list
        The pyCGNS Base node (CGNS/SIDS tree node).
    X, Y, Z : np.ndarray
        Vertex grids of shape ``(n_sections, m_points)``.
    zone_name : str
        CGNS zone name.
    closed : bool
        Include end-cap triangles for watertight mesh.
    """
    import CGNS.PAT.cgnslib as CGL
    import CGNS.PAT.cgnskeywords as CGK

    coords, faces = _build_tri_connectivity(X, Y, Z, closed=closed)
    n_verts = coords.shape[0]
    n_tris = faces.shape[0]

    # ── Zone node ─────────────────────────────────────────────
    # ZoneSize: [NVertex, NCell, NBoundVertex=0]
    zone_size = np.array([[n_verts, n_tris, 0]], dtype=np.int32)
    zone = CGL.newZone(base_node, zone_name, zone_size, CGK.Unstructured_s)

    # ── GridCoordinates ───────────────────────────────────────
    gc = CGL.newGridCoordinates(zone, "GridCoordinates")
    CGL.newDataArray(gc, "CoordinateX", coords[:, 0].astype(np.float64).copy())
    CGL.newDataArray(gc, "CoordinateY", coords[:, 1].astype(np.float64).copy())
    CGL.newDataArray(gc, "CoordinateZ", coords[:, 2].astype(np.float64).copy())

    # ── Elements (TRI_3) ──────────────────────────────────────
    # Convert 0-based to 1-based connectivity and flatten
    connectivity = (faces + 1).flatten().astype(np.int32)
    erange = np.array([1, n_tris], dtype=np.int32)
    CGL.newElements(zone, "TriElements", CGK.TRI_3,
                    erange, connectivity)


