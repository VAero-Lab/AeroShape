"""Aerodynamic profile generation and 3D wing mesh construction.

This module provides tools for generating NACA 4-digit airfoil profiles
and constructing 3D wing meshes by interpolating between root and tip
profiles along the span. The geometry is represented as structured vertex
matrices Vk (Eq. 1 in the paper), which serve as input for the volume
and mass computation routines.

Reference:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378.
"""

import numpy as np
import math


class NACAProfileGenerator:
    """Generates 4-digit NACA airfoil profiles.

    Produces the (x, z) coordinates of an airfoil cross-section using
    the standard NACA 4-digit series equations. A cosine spacing
    distribution is used to cluster more points near the leading and
    trailing edges, as recommended in Section 2.1 of the paper (Fig. 2c).
    """

    @staticmethod
    def generate(naca_code, num_points=50, chord=1.0):
        """Generate the (x, z) coordinates of a 4-digit NACA profile.

        Parameters
        ----------
        naca_code : str or int
            Four-digit NACA designation (e.g. '2412').
        num_points : int
            Number of points per surface (upper or lower). Total profile
            points will be approximately 2 * num_points - 1.
        chord : float
            Chord length in meters.

        Returns
        -------
        x : np.ndarray
            Chordwise coordinates, ordered from lower-surface trailing edge
            through the leading edge to upper-surface trailing edge.
        z : np.ndarray
            Thickness coordinates corresponding to x.
        """
        naca_code = str(naca_code).zfill(4)
        m = int(naca_code[0]) / 100.0   # Maximum camber
        p = int(naca_code[1]) / 10.0    # Position of maximum camber
        t = int(naca_code[2:]) / 100.0  # Maximum thickness

        # Cosine spacing for better LE/TE resolution (Fig. 2c)
        beta = np.linspace(0.0, math.pi, num_points)
        xc = 0.5 * (1.0 - np.cos(beta))

        # Thickness distribution (standard NACA formula)
        yt = 5.0 * t * (
            0.2969 * np.sqrt(xc)
            - 0.1260 * xc
            - 0.3516 * xc**2
            + 0.2843 * xc**3
            - 0.1015 * xc**4
        )

        # Camber line and its derivative
        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)
        if m > 0.0:
            for i in range(len(xc)):
                if xc[i] < p:
                    yc[i] = (m / p**2) * (2.0 * p * xc[i] - xc[i]**2)
                    dyc_dx[i] = (2.0 * m / p**2) * (p - xc[i])
                else:
                    yc[i] = (m / (1.0 - p)**2) * (
                        (1.0 - 2.0 * p) + 2.0 * p * xc[i] - xc[i]**2
                    )
                    dyc_dx[i] = (2.0 * m / (1.0 - p)**2) * (p - xc[i])

        theta = np.arctan(dyc_dx)

        # Upper and lower surface coordinates
        xu = xc - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = xc + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Concatenate: lower surface (reversed) + upper surface
        x = np.concatenate((xl[::-1], xu[1:])) * chord
        z = np.concatenate((yl[::-1], yu[1:])) * chord
        return x, z


class WingMeshFactory:
    """Constructs 3D wing meshes from root and tip airfoil profiles.

    Creates the structured vertex representation Vk (Eq. 1) by linearly
    interpolating between root and tip NACA profiles along the spanwise
    direction, with optional sweep angle transformation.
    """

    @staticmethod
    def create(naca_root, naca_tip, semi_span, chord_root, chord_tip,
               sweep_angle_deg, num_points_profile=20, num_sections=10):
        """Create the (X, Y, Z) coordinate matrices for a 3D wing.

        Parameters
        ----------
        naca_root : str or int
            NACA 4-digit code for the root airfoil.
        naca_tip : str or int
            NACA 4-digit code for the tip airfoil.
        semi_span : float
            Wing semi-span in meters.
        chord_root : float
            Root chord length in meters.
        chord_tip : float
            Tip chord length in meters.
        sweep_angle_deg : float
            Leading-edge sweep angle in degrees.
        num_points_profile : int
            Number of points per airfoil profile (chordwise, parameter m).
        num_sections : int
            Number of spanwise sections (parameter n).

        Returns
        -------
        X, Y, Z : np.ndarray
            Coordinate matrices of shape (num_sections, num_profile_points).
            These form the vertex representation Vk (Eq. 1).
        """
        x_root, z_root = NACAProfileGenerator.generate(
            naca_root, num_points_profile, chord_root
        )
        x_tip, z_tip = NACAProfileGenerator.generate(
            naca_tip, num_points_profile, chord_tip
        )

        y = np.linspace(0, semi_span, num_sections)

        sweep_rad = math.radians(sweep_angle_deg)
        x_offset_tip = semi_span * math.tan(sweep_rad)

        X = np.zeros((num_sections, len(x_root)))
        Y = np.zeros((num_sections, len(x_root)))
        Z = np.zeros((num_sections, len(x_root)))

        for i in range(num_sections):
            t = i / (num_sections - 1.0)
            x_interp = (1 - t) * x_root + t * x_tip
            z_interp = (1 - t) * z_root + t * z_tip
            x_offset = t * x_offset_tip

            X[i, :] = x_interp + x_offset
            Y[i, :] = y[i]
            Z[i, :] = z_interp

        return X, Y, Z
