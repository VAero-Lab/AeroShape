"""Airfoil profile generation and representation.

Provides the AirfoilProfile class that unifies NACA parametric profiles,
custom coordinate files (.dat), and arbitrary point arrays into a single
object that can produce both numpy arrays and OpenCASCADE B-spline wires
suitable for NURBS lofting.

Coordinate convention:
    Points are ordered lower-surface trailing edge -> leading edge ->
    upper-surface trailing edge, matching NACAProfileGenerator output.
    Coordinates are in the XZ plane (X = chordwise, Z = thickness).
"""

import math
from dataclasses import dataclass
import numpy as np


@dataclass
class AirfoilProfile:
    """A 2D airfoil profile stored as coordinate arrays.

    Attributes
    ----------
    x : np.ndarray
        Chordwise coordinates.
    z : np.ndarray
        Thickness coordinates.
    name : str
        Descriptive name.
    chord : float
        Reference chord length these coordinates were generated at.
    """
    x: np.ndarray
    z: np.ndarray
    name: str = ""
    chord: float = 1.0

    # ── Factory methods ───────────────────────────────────────────

    @classmethod
    def from_naca4(cls, code, num_points=50, chord=1.0):
        """Create from a NACA 4-digit code.

        Delegates to the existing NACAProfileGenerator.
        """
        from aeroshape.geometry import NACAProfileGenerator
        x, z = NACAProfileGenerator.generate(str(code), num_points, chord)
        return cls(x=x, z=z, name=f"NACA {code}", chord=chord)

    @classmethod
    def from_naca5(cls, code, num_points=50, chord=1.0):
        """Create from a NACA 5-digit code.

        Implements the standard 5-digit series:
        - First digit * 3/20 = design lift coefficient
        - Next two digits / 2 = position of max camber (% chord * 10)
        - Last two digits = max thickness (% chord)
        """
        code = str(code).zfill(5)
        cl_design = int(code[0]) * 3.0 / 20.0
        p = int(code[1:3]) / 200.0
        t_max = int(code[3:5]) / 100.0

        # Cosine spacing
        beta = np.linspace(0, np.pi, num_points)
        xc = 0.5 * (1.0 - np.cos(beta))

        # Thickness distribution (same as 4-digit)
        yt = 5.0 * t_max * (
            0.2969 * np.sqrt(xc)
            - 0.1260 * xc
            - 0.3516 * xc**2
            + 0.2843 * xc**3
            - 0.1015 * xc**4
        )

        # 5-digit mean camber line coefficients
        if p > 0:
            m = cl_design
            k1_table = {
                0.05: 361.400, 0.10: 51.640, 0.15: 15.957,
                0.20: 6.643, 0.25: 3.230,
            }
            # Find closest p in table
            p_key = min(k1_table.keys(), key=lambda pk: abs(pk - p))
            k1 = k1_table[p_key]

            yc = np.zeros_like(xc)
            dyc_dx = np.zeros_like(xc)
            for i in range(len(xc)):
                if xc[i] < p:
                    yc[i] = (k1 / 6.0) * (
                        xc[i]**3 - 3.0 * p * xc[i]**2
                        + p**2 * (3.0 - p) * xc[i]
                    )
                    dyc_dx[i] = (k1 / 6.0) * (
                        3.0 * xc[i]**2 - 6.0 * p * xc[i]
                        + p**2 * (3.0 - p)
                    )
                else:
                    yc[i] = (k1 * p**3 / 6.0) * (1.0 - xc[i])
                    dyc_dx[i] = -(k1 * p**3 / 6.0)
        else:
            yc = np.zeros_like(xc)
            dyc_dx = np.zeros_like(xc)

        theta = np.arctan(dyc_dx)

        xu = xc - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = xc + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        x = np.concatenate((xl[::-1], xu[1:])) * chord
        z = np.concatenate((yl[::-1], yu[1:])) * chord
        return cls(x=x, z=z, name=f"NACA {code}", chord=chord)

    @classmethod
    def from_dat_file(cls, filepath, chord=1.0):
        """Load from a Selig or Lednicer format .dat file.

        Handles both formats automatically:
        - Selig: contiguous (x, z) pairs from upper TE around LE to lower TE
        - Lednicer: header with point counts, then upper and lower blocks
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        name = lines[0].strip()

        # Parse numeric data
        coords = []
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

        if not coords:
            raise ValueError(f"No coordinate data found in {filepath}")

        coords = np.array(coords)

        # Detect Lednicer format: first data line has two integers
        first_line = lines[1].strip().split()
        is_lednicer = False
        if len(first_line) == 2:
            try:
                n1, n2 = int(float(first_line[0])), int(float(first_line[1]))
                if n1 + n2 == len(coords) or n1 + n2 == len(coords) + 1:
                    is_lednicer = True
            except (ValueError, IndexError):
                pass

        if is_lednicer:
            # Lednicer: upper surface then lower surface, both LE to TE
            # Need to combine into: lower TE → LE → upper TE
            upper = coords[:len(coords) // 2]
            lower = coords[len(coords) // 2:]
            x = np.concatenate((lower[::-1, 0], upper[1:, 0]))
            z = np.concatenate((lower[::-1, 1], upper[1:, 1]))
        else:
            # Selig: upper TE → LE → lower TE (already contiguous)
            # Reverse to match our convention: lower TE → LE → upper TE
            x = coords[::-1, 0]
            z = coords[::-1, 1]

        # Normalize to unit chord then scale
        x_range = x.max() - x.min()
        if x_range > 0:
            x = (x - x.min()) / x_range * chord
            z = z / x_range * chord

        return cls(x=x, z=z, name=name, chord=chord)

    @classmethod
    def from_points(cls, x, z, name="custom", chord=1.0):
        """Create from explicit coordinate arrays."""
        return cls(
            x=np.asarray(x, dtype=float),
            z=np.asarray(z, dtype=float),
            name=name,
            chord=chord,
        )

    # ── Transformations ───────────────────────────────────────────

    def scaled(self, new_chord):
        """Return a copy scaled to a different chord length."""
        factor = new_chord / self.chord
        return AirfoilProfile(
            x=self.x * factor,
            z=self.z * factor,
            name=self.name,
            chord=new_chord,
        )

    # ── OCC conversion ────────────────────────────────────────────

    def to_occ_wire(self, position=(0.0, 0.0, 0.0), twist_deg=0.0,
                    local_chord=None):
        """Convert to a pythonocc TopoDS_Wire at a 3D position.

        The profile is placed in the XZ plane, centered at the leading
        edge, then twisted about the LE, scaled to local_chord, and
        translated to position.

        Parameters
        ----------
        position : tuple
            (x_offset, y_position, z_offset) in meters.
        twist_deg : float
            Twist angle about the leading edge (degrees).
        local_chord : float or None
            If given, scale the profile to this chord before placing.

        Returns
        -------
        TopoDS_Wire
            Closed B-spline wire suitable for lofting.
        """
        from OCP.gp import gp_Pnt
        from OCP.GeomAPI import GeomAPI_PointsToBSpline
        from OCP.GeomAbs import GeomAbs_C2
        from OCP.TColgp import TColgp_Array1OfPnt
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
        )

        # Scale to local chord if needed
        if local_chord is not None and abs(local_chord - self.chord) > 1e-10:
            profile = self.scaled(local_chord)
        else:
            profile = self

        px, pz = profile.x.copy(), profile.z.copy()

        # Apply twist about the leading edge (minimum x point)
        if abs(twist_deg) > 1e-10:
            le_idx = np.argmin(px)
            le_x, le_z = px[le_idx], pz[le_idx]
            angle = math.radians(twist_deg)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            dx = px - le_x
            dz = pz - le_z
            px = le_x + dx * cos_a + dz * sin_a
            pz = le_z - dx * sin_a + dz * cos_a

        # Translate to position
        x_off, y_pos, z_off = position
        n = len(px)
        arr = TColgp_Array1OfPnt(1, n)
        for i in range(n):
            arr.SetValue(i + 1, gp_Pnt(
                float(px[i]) + x_off,
                float(y_pos),
                float(pz[i]) + z_off,
            ))

        # Fit B-spline curve through points
        bspline = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-4)
        if not bspline.IsDone():
            bspline = GeomAPI_PointsToBSpline(arr)

        edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        return wire
