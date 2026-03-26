"""Volume computation for lifting surfaces and thin-shell structures.

This module implements the volume computation routines of the GVM
methodology, based on the discrete form of the Divergence Theorem.

Supported structural topologies (Fig. 1 in the paper):
- Solid body: direct volume from a watertight triangulated surface
- Thin-shell (Approach I - Offset): Volume(shell) = Volume(outer) - Volume(inner)
- Thin-shell (Approach II - Unfolding): Volume(shell) = wetted_area * t_shell

The surface area computation (wetted area) is also provided, as it is
required for the unfolding approach and is generally useful.

Reference:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378, Section 2.2.
"""

import numpy as np


class VolumeCalculator:
    """Computes volume and surface area of triangulated lifting surfaces.

    Provides methods for solid body and thin-shell volume computation
    using the Divergence Theorem (Eq. 2) and the unfolding approximation.
    """

    @staticmethod
    def compute_solid_volume(triangles):
        """Compute the volume enclosed by a watertight triangulated surface.

        Uses the discrete form of the Divergence Theorem (Eq. 2):

            Volume(Vk) = |1/3 * sum( Q_ABC . N_ABC * area(ABC) )|

        where Q_ABC is the facet centroid, N_ABC is the facet normal,
        and area(ABC) is the facet area.

        The triangulated surface must be watertight (closed) for the
        result to be physically meaningful. Use MeshTopologyManager with
        closed=True, or get_thick_shell_triangles for thin-shell meshes.

        Parameters
        ----------
        triangles : list of tuple
            List of (A, B, C) triangular facets, where each vertex is
            a np.ndarray of shape (3,).

        Returns
        -------
        float
            Volume of the enclosed region in cubic meters.
        """
        total_volume = 0.0
        for A, B, C in triangles:
            # Area vector = cross(AB, AC) / 2 = N_ABC * area(ABC) / 2
            area_vec = np.cross((B - A), (C - A)) / 2.0
            # Centroid Q_ABC = (A + B + C) / 3
            centroid = (A + B + C) / 3.0
            # Divergence Theorem contribution: Q . (N * area)
            total_volume += np.dot(centroid, area_vec)
        return abs(total_volume / 3.0)

    @staticmethod
    def compute_shell_volume_offset(X, Y, Z, t_shell):
        """Compute thin-shell volume using the offset approach (Approach I).

        Offsets the outer geometry inward by t_shell along vertex normals
        to create an inner boundary, then computes:

            Volume(shell) = Volume(outer) - Volume(inner)

        Both volumes are computed independently via the Divergence Theorem
        on watertight (closed) triangulations (Fig. 4 left in the paper).

        Parameters
        ----------
        X, Y, Z : np.ndarray
            Outer surface coordinate matrices (n_sections x m_points).
        t_shell : float
            Shell/skin thickness in meters.

        Returns
        -------
        volume_shell : float
            Shell volume in cubic meters.
        volume_outer : float
            Volume of the outer boundary in cubic meters.
        volume_inner : float
            Volume of the inner boundary in cubic meters.

        See Also
        --------
        compute_shell_volume_unfolding : Simpler approximation method.
        """
        from aeroshape.mesh_utils import MeshTopologyManager

        n_sec, m_pts = X.shape

        # --- Compute vertex normals by averaging adjacent face normals ---
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

        # --- Inner surface: offset inward by shell thickness ---
        Xi = X - Nx * t_shell
        Yi = Y - Ny * t_shell
        Zi = Z - Nz * t_shell

        # --- Compute volumes of outer and inner boundaries independently ---
        outer_triangles = MeshTopologyManager.get_wing_triangles(
            X, Y, Z, closed=True
        )
        inner_triangles = MeshTopologyManager.get_wing_triangles(
            Xi, Yi, Zi, closed=True
        )

        volume_outer = VolumeCalculator.compute_solid_volume(outer_triangles)
        volume_inner = VolumeCalculator.compute_solid_volume(inner_triangles)
        volume_shell = volume_outer - volume_inner

        return volume_shell, volume_outer, volume_inner

    @staticmethod
    def compute_shell_volume_unfolding(triangles, t_shell):
        """Compute thin-shell volume using the unfolding approach (Approach II).

        A simplified method that "unfolds" the lifting surface and computes
        the volume as the product of wetted area and shell thickness
        (Fig. 4 right in the paper):

            Volume(shell) = area(Vk)_wetted * t_shell

        This approach is simpler but slightly less accurate than the offset
        method. The difference is typically small for thin shells.

        Parameters
        ----------
        triangles : list of tuple
            Triangulated outer surface (from get_wing_triangles with
            closed=True).
        t_shell : float
            Shell/skin thickness in meters.

        Returns
        -------
        float
            Approximate shell volume in cubic meters.

        See Also
        --------
        compute_shell_volume_offset : More accurate offset method.
        """
        wetted_area = VolumeCalculator.compute_surface_area(triangles)
        return wetted_area * t_shell

    @staticmethod
    def compute_surface_area(triangles):
        """Compute the total surface area of a triangulated mesh.

        Sums the area of all triangular facets:
            area(Vk)_wetted = sum( area(ABC) )

        where area(ABC) = |cross(AB, AC)| / 2.

        Parameters
        ----------
        triangles : list of tuple
            List of (A, B, C) triangular facets.

        Returns
        -------
        float
            Total surface area in square meters.
        """
        total_area = 0.0
        for A, B, C in triangles:
            total_area += np.linalg.norm(np.cross(B - A, C - A)) / 2.0
        return total_area

    @staticmethod
    def compute_facet_properties(A, B, C):
        """Compute the centroid, unit normal, and area of a single triangle.

        Parameters
        ----------
        A, B, C : np.ndarray
            Triangle vertices, each of shape (3,).

        Returns
        -------
        centroid : np.ndarray
            Centroid (A + B + C) / 3, shape (3,).
        normal : np.ndarray
            Unit normal vector, shape (3,). Zero vector if degenerate.
        area : float
            Triangle area in square meters.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)

        cross = np.cross(B - A, C - A)
        area = np.linalg.norm(cross) / 2.0

        if area > 1e-10:
            normal = cross / np.linalg.norm(cross)
        else:
            normal = np.array([0.0, 0.0, 0.0])

        centroid = (A + B + C) / 3.0
        return centroid, normal, area
