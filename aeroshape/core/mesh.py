"""Triangular mesh generation for lifting surfaces.

This module implements the structured triangulation approach described
in Section 2.2.1 of the paper. It generates watertight triangulated
meshes from the vertex representation Vk, supporting both solid body
and thin-shell (offset approach) topologies.

The triangulation follows the paper's convention:
- Outer surface: pairs of triangles (ABC_left, ABC_right) per quad (Table 1)
- Cross sections: fan triangulation at root and tip end-caps
- Thin-shell: double-wall mesh via normal-offset (Fig. 4, Approach I)

Normal vectors point inward to satisfy the Divergence Theorem convention.

Reference:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378.
"""

import numpy as np


class MeshTopologyManager:
    """Generates triangulated meshes from structured vertex grids.

    Handles the conversion of the vertex representation Vk (m x n matrix)
    into a list of triangular facets suitable for volume computation via
    the Divergence Theorem (Eq. 2).
    """

    @staticmethod
    def get_wing_triangles(X, Y, Z, closed=True):
        """Triangulate the lifting surface from its vertex grid.

        Implements the structured triangulation of Section 2.2.1 (Fig. 3).
        Iterates over the vertex matrix Vk in the chordwise (i) and
        spanwise (j) directions, generating triangle pairs per quad cell.

        Parameters
        ----------
        X, Y, Z : np.ndarray
            Coordinate matrices of shape (n_sections, m_points), forming
            the vertex representation Vk.
        closed : bool
            If True, triangulate the root and tip cross-section end-caps
            to create a watertight volume for the Divergence Theorem.

        Returns
        -------
        triangles : list of tuple
            Each element is (A, B, C) where A, B, C are np.ndarray(3,)
            vertices defining a triangular facet.
        """
        triangles = []
        num_sections = X.shape[0]
        num_points = X.shape[1]

        # Triangulation of the outer surface (Fig. 3, left)
        # Iterates spanwise (j) and chordwise (i), creating quad -> 2 triangles
        for j in range(num_sections - 1):
            for i in range(num_points - 1):
                A = np.array([X[j, i], Y[j, i], Z[j, i]])
                B = np.array([X[j, i+1], Y[j, i+1], Z[j, i+1]])
                C = np.array([X[j+1, i], Y[j+1, i], Z[j+1, i]])
                D = np.array([X[j+1, i+1], Y[j+1, i+1], Z[j+1, i+1]])
                triangles.append((A, B, C))
                triangles.append((B, D, C))

        # Close the trailing-edge gap (connect first and last chordwise points)
        for j in range(num_sections - 1):
            A = np.array([X[j, 0], Y[j, 0], Z[j, 0]])
            C = np.array([X[j+1, 0], Y[j+1, 0], Z[j+1, 0]])
            B = np.array([X[j, num_points-1], Y[j, num_points-1],
                          Z[j, num_points-1]])
            D = np.array([X[j+1, num_points-1], Y[j+1, num_points-1],
                          Z[j+1, num_points-1]])
            if np.linalg.norm(A - B) > 1e-6:
                n1 = np.cross(B - A, C - A)
                if n1[0] < 0:
                    triangles.append((A, C, B))
                else:
                    triangles.append((A, B, C))
                n2 = np.cross(D - B, C - B)
                if n2[0] < 0:
                    triangles.append((B, C, D))
                else:
                    triangles.append((B, D, C))

        if closed:
            # Triangulation of the root cross section (Fig. 3, right-top)
            center_root = np.array([
                np.mean(X[0, :]), np.mean(Y[0, :]), np.mean(Z[0, :])
            ])
            for i in range(num_points - 1):
                A = np.array([X[0, i], Y[0, i], Z[0, i]])
                B = np.array([X[0, i+1], Y[0, i+1], Z[0, i+1]])
                ntest = np.cross(B - center_root, A - center_root)
                if ntest[1] < 0:
                    triangles.append((center_root, B, A))
                else:
                    triangles.append((center_root, A, B))

            # Close the root trailing-edge gap
            A = np.array([X[0, num_points-1], Y[0, num_points-1],
                          Z[0, num_points-1]])
            B = np.array([X[0, 0], Y[0, 0], Z[0, 0]])
            if np.linalg.norm(A - B) > 1e-6:
                ntest = np.cross(B - center_root, A - center_root)
                if ntest[1] < 0:
                    triangles.append((center_root, B, A))
                else:
                    triangles.append((center_root, A, B))

            # Triangulation of the tip cross section
            j_tip = num_sections - 1
            center_tip = np.array([
                np.mean(X[j_tip, :]), np.mean(Y[j_tip, :]),
                np.mean(Z[j_tip, :])
            ])
            for i in range(num_points - 1):
                A = np.array([X[j_tip, i], Y[j_tip, i], Z[j_tip, i]])
                B = np.array([X[j_tip, i+1], Y[j_tip, i+1],
                              Z[j_tip, i+1]])
                ntest = np.cross(B - center_tip, A - center_tip)
                if ntest[1] > 0:
                    triangles.append((center_tip, B, A))
                else:
                    triangles.append((center_tip, A, B))

            # Close the tip trailing-edge gap
            A = np.array([X[j_tip, num_points-1], Y[j_tip, num_points-1],
                          Z[j_tip, num_points-1]])
            B = np.array([X[j_tip, 0], Y[j_tip, 0], Z[j_tip, 0]])
            if np.linalg.norm(A - B) > 1e-6:
                ntest = np.cross(B - center_tip, A - center_tip)
                if ntest[1] > 0:
                    triangles.append((center_tip, B, A))
                else:
                    triangles.append((center_tip, A, B))

        return triangles

    @staticmethod
    def get_thick_shell_triangles(X, Y, Z, t_shell):
        """Create a watertight thin-shell mesh using the offset approach.

        Implements the offset approach for thin-shell volume computation
        (Section 2.2.1, Fig. 4 left). Creates an inner surface by offsetting
        each vertex inward along its averaged normal by t_shell, then builds
        a double-walled hermetic mesh: outer surface + reversed inner surface
        + connecting quads at root and tip.

        Volume(shell) = Volume(outer) - Volume(inner), computed automatically
        by the Divergence Theorem on the combined watertight mesh.

        Parameters
        ----------
        X, Y, Z : np.ndarray
            Outer surface coordinate matrices (n_sections x m_points).
        t_shell : float
            Shell/skin thickness in meters.

        Returns
        -------
        triangles : list of tuple
            Watertight triangulated mesh of the thin-shell volume.
        """
        n_sec, m_pts = X.shape

        # Compute vertex normals by averaging adjacent face normals
        Nx = np.zeros_like(X)
        Ny = np.zeros_like(Y)
        Nz = np.zeros_like(Z)

        for j in range(n_sec - 1):
            for i in range(m_pts - 1):
                A = np.array([X[j, i], Y[j, i], Z[j, i]])
                B = np.array([X[j, i+1], Y[j, i+1], Z[j, i+1]])
                C = np.array([X[j+1, i], Y[j+1, i], Z[j+1, i]])
                D = np.array([X[j+1, i+1], Y[j+1, i+1], Z[j+1, i+1]])
                n1 = np.cross(B - A, C - A)
                n2 = np.cross(D - B, C - B)
                Nx[j, i] += n1[0]
                Ny[j, i] += n1[1]
                Nz[j, i] += n1[2]
                Nx[j, i+1] += n1[0] + n2[0]
                Ny[j, i+1] += n1[1] + n2[1]
                Nz[j, i+1] += n1[2] + n2[2]
                Nx[j+1, i] += n1[0] + n2[0]
                Ny[j+1, i] += n1[1] + n2[1]
                Nz[j+1, i] += n1[2] + n2[2]
                Nx[j+1, i+1] += n2[0]
                Ny[j+1, i+1] += n2[1]
                Nz[j+1, i+1] += n2[2]

        # Normalize vertex normals
        for j in range(n_sec):
            for i in range(m_pts):
                norm = np.sqrt(Nx[j, i]**2 + Ny[j, i]**2 + Nz[j, i]**2)
                if norm > 1e-8:
                    Nx[j, i] /= norm
                    Ny[j, i] /= norm
                    Nz[j, i] /= norm

        # Inner surface: offset inward by shell thickness
        Xi = X - Nx * t_shell
        Yi = Y - Ny * t_shell
        Zi = Z - Nz * t_shell

        # Outer surface triangles (open, no end-caps)
        triangles = MeshTopologyManager.get_wing_triangles(
            X, Y, Z, closed=False
        )

        # Inner surface triangles (reversed winding for inward normals)
        t_in = MeshTopologyManager.get_wing_triangles(
            Xi, Yi, Zi, closed=False
        )
        triangles.extend([(C, B, A) for A, B, C in t_in])

        def add_quad(P1, P2, P3, P4, n_exp):
            """Add a quad (as 2 triangles) with correct winding."""
            n = np.cross(P2 - P1, P3 - P1)
            if np.dot(n, n_exp) < 0:
                triangles.extend([(P1, P3, P2), (P1, P4, P3)])
            else:
                triangles.extend([(P1, P2, P3), (P1, P3, P4)])

        # Connect outer and inner surfaces at root (j=0)
        for i in range(m_pts - 1):
            P1 = np.array([X[0, i], Y[0, i], Z[0, i]])
            P2 = np.array([X[0, i+1], Y[0, i+1], Z[0, i+1]])
            P3 = np.array([Xi[0, i+1], Yi[0, i+1], Zi[0, i+1]])
            P4 = np.array([Xi[0, i], Yi[0, i], Zi[0, i]])
            add_quad(P1, P2, P3, P4, np.array([0, -1, 0]))

        # Close root trailing-edge connection
        root_le = np.array([X[0, 0], Y[0, 0], Z[0, 0]])
        root_te = np.array([X[0, m_pts-1], Y[0, m_pts-1], Z[0, m_pts-1]])
        if np.linalg.norm(root_le - root_te) > 1e-6:
            P1 = root_te
            P2 = root_le
            P3 = np.array([Xi[0, 0], Yi[0, 0], Zi[0, 0]])
            P4 = np.array([Xi[0, m_pts-1], Yi[0, m_pts-1], Zi[0, m_pts-1]])
            add_quad(P1, P2, P3, P4, np.array([0, -1, 0]))

        # Connect outer and inner surfaces at tip (j=n_sec-1)
        j = n_sec - 1
        for i in range(m_pts - 1):
            P1 = np.array([X[j, i], Y[j, i], Z[j, i]])
            P2 = np.array([X[j, i+1], Y[j, i+1], Z[j, i+1]])
            P3 = np.array([Xi[j, i+1], Yi[j, i+1], Zi[j, i+1]])
            P4 = np.array([Xi[j, i], Yi[j, i], Zi[j, i]])
            add_quad(P1, P2, P3, P4, np.array([0, 1, 0]))

        # Close tip trailing-edge connection
        tip_le = np.array([X[j, 0], Y[j, 0], Z[j, 0]])
        tip_te = np.array([X[j, m_pts-1], Y[j, m_pts-1], Z[j, m_pts-1]])
        if np.linalg.norm(tip_le - tip_te) > 1e-6:
            P1 = tip_te
            P2 = tip_le
            P3 = np.array([Xi[j, 0], Yi[j, 0], Zi[j, 0]])
            P4 = np.array([Xi[j, m_pts-1], Yi[j, m_pts-1], Zi[j, m_pts-1]])
            add_quad(P1, P2, P3, P4, np.array([0, 1, 0]))

        return triangles
