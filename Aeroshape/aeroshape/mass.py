"""Mass distribution and inertial properties for lifting surfaces.

This module implements the mass properties computation routines of the
GVM methodology, based on a system of mass particles derived from the
three-dimensional distribution of chord and thickness.

The mass distribution model (Section 2.3.1) uses two key geometric
features to distribute mass throughout the lifting surface:
- Spanwise: chord and maximum thickness distribution (Eqs. 4-6)
- Chordwise: local airfoil thickness distribution (Eqs. 7-8)

The resulting 3D mass distribution matrix (Eq. 9) is used to compute
the center of mass (Eq. 10-11) and the full inertia tensor (Eqs. 12-14).

Reference:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378, Section 2.3.
"""

import numpy as np


class MassPropertiesCalculator:
    """Computes mass distribution and inertial properties of lifting surfaces.

    Uses a particle-based approach where each vertex in the surface mesh
    is assigned a mass based on the local chord and thickness distributions.
    """

    @staticmethod
    def compute_mass_distribution(X, Y, Z, total_mass):
        """Build the 3D mass distribution matrix M_3D_dist (Eq. 9).

        Distributes the total mass across the vertex grid using the
        chord-thickness model:

        1. Chord distribution C_dist (Eq. 4) — normalized chord at each
           spanwise section.
        2. Thickness distribution T_dist (Eq. 5) — normalized max thickness
           at each spanwise section.
        3. Strip mass M_strip_j = C_j * T_j * M_ratio (Eq. 6).
        4. Local thickness t_dist_j (Eq. 7) — chordwise distribution within
           each strip.
        5. Particle mass m_dot_ij = t_ij * m_ratio (Eq. 8).

        Parameters
        ----------
        X, Y, Z : np.ndarray
            Vertex coordinate matrices of shape (n_sections, m_points).
        total_mass : float
            Total mass of the lifting surface in kg.

        Returns
        -------
        M_3D_dist : np.ndarray
            Mass distribution matrix of shape (n_sections, m_points),
            where each element is the mass assigned to that vertex.
        M_strip : np.ndarray
            Strip mass array of shape (n_sections,), the total mass
            assigned to each spanwise section.
        """
        n_sec = X.shape[0]
        m_pts = X.shape[1]

        # Step 1: Chord and thickness distributions (Eqs. 4-5)
        C = np.zeros(n_sec)
        T = np.zeros(n_sec)
        for j in range(n_sec):
            C[j] = np.max(X[j, :]) - np.min(X[j, :])
            T[j] = np.max(Z[j, :]) - np.min(Z[j, :])

        C_dist = C / np.max(C) if np.max(C) > 0 else C
        T_dist = T / np.max(T) if np.max(T) > 0 else T

        # Step 2: Strip mass via chord-thickness factor (Eq. 6)
        factor_ct = C_dist * T_dist
        sum_factor = np.sum(factor_ct)
        M_ratio = total_mass / sum_factor if sum_factor > 0 else 0
        M_strip = factor_ct * M_ratio

        # Steps 3-5: Local thickness distribution and particle masses (Eqs. 7-9)
        M_3D_dist = np.zeros((n_sec, m_pts))
        for j in range(n_sec):
            # Local airfoil thickness at each chordwise point
            T_local = np.abs(Z[j, :] - np.mean(Z[j, :]))
            max_T_loc = np.max(T_local)

            if max_T_loc < 1e-6:
                t_dist = np.ones(m_pts)
            else:
                t_dist = T_local / max_T_loc

            sum_t = np.sum(t_dist)
            m_ratio = M_strip[j] / sum_t if sum_t > 0 else 0
            M_3D_dist[j, :] = t_dist * m_ratio

        return M_3D_dist, M_strip

    @staticmethod
    def compute_center_of_mass(X, Y, Z, M_3D_dist, total_mass):
        """Compute the center of mass of the lifting surface (Eq. 10).

        Uses the particle-system definition:
            x_com = (1/M) * sum_ij( X_ij * m_ij )

        Parameters
        ----------
        X, Y, Z : np.ndarray
            Vertex coordinate matrices of shape (n_sections, m_points).
        M_3D_dist : np.ndarray
            Mass distribution matrix from compute_mass_distribution().
        total_mass : float
            Total mass of the lifting surface in kg.

        Returns
        -------
        tuple of float
            (CG_X, CG_Y, CG_Z) center of mass coordinates in meters.
        """
        if total_mass > 0:
            CG_X = np.sum(X * M_3D_dist) / total_mass
            CG_Y = np.sum(Y * M_3D_dist) / total_mass
            CG_Z = np.sum(Z * M_3D_dist) / total_mass
        else:
            CG_X, CG_Y, CG_Z = 0.0, 0.0, 0.0

        return (CG_X, CG_Y, CG_Z)

    @staticmethod
    def compute_inertia_tensor(X, Y, Z, M_3D_dist):
        """Compute the inertia tensor of the lifting surface (Eqs. 12-13).

        Computes all six independent components of the symmetric inertia
        tensor using the particle-system formulation:
            Ixx = sum( m_ij * (Y_ij^2 + Z_ij^2) )
            Ixy = -sum( m_ij * X_ij * Y_ij )
            etc.

        Parameters
        ----------
        X, Y, Z : np.ndarray
            Vertex coordinate matrices of shape (n_sections, m_points).
        M_3D_dist : np.ndarray
            Mass distribution matrix from compute_mass_distribution().

        Returns
        -------
        tuple of float
            (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) inertia tensor components
            in kg*m^2.
        """
        Ixx = np.sum(M_3D_dist * (Y**2 + Z**2))
        Iyy = np.sum(M_3D_dist * (X**2 + Z**2))
        Izz = np.sum(M_3D_dist * (X**2 + Y**2))

        Ixy = -np.sum(M_3D_dist * X * Y)
        Ixz = -np.sum(M_3D_dist * X * Z)
        Iyz = -np.sum(M_3D_dist * Y * Z)

        return (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

    @staticmethod
    def compute_all(X, Y, Z, total_mass):
        """Compute mass distribution, center of mass, and inertia tensor.

        Convenience method that calls compute_mass_distribution,
        compute_center_of_mass, and compute_inertia_tensor in sequence.

        Parameters
        ----------
        X, Y, Z : np.ndarray
            Vertex coordinate matrices of shape (n_sections, m_points).
        total_mass : float
            Total mass of the lifting surface in kg.

        Returns
        -------
        center_of_mass : tuple of float
            (CG_X, CG_Y, CG_Z) center of mass coordinates in meters.
        inertia : tuple of float
            (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) inertia tensor components
            in kg*m^2.
        mass_distribution : np.ndarray
            3D mass distribution matrix M_3D_dist (Eq. 9).
        """
        M_3D_dist, _ = MassPropertiesCalculator.compute_mass_distribution(
            X, Y, Z, total_mass
        )
        center_of_mass = MassPropertiesCalculator.compute_center_of_mass(
            X, Y, Z, M_3D_dist, total_mass
        )
        inertia = MassPropertiesCalculator.compute_inertia_tensor(
            X, Y, Z, M_3D_dist
        )
        return center_of_mass, inertia, M_3D_dist
