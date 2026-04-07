"""Airfoil profile generation and representation.

Provides NACAProfileGenerator for parametric NACA profiles and the
AirfoilProfile class that unifies NACA (4 & 5-digit), .dat file, and
arbitrary point arrays into a single object.  Profiles can produce both
numpy arrays and OpenCASCADE B-spline wires for NURBS lofting.

Coordinate convention:
    Points are ordered lower-surface trailing edge -> leading edge ->
    upper-surface trailing edge.  Coordinates are in the XZ plane
    (X = chordwise, Z = thickness).

Reference:
    Valencia et al., "A CAD-free methodology for volume and mass properties
    computation of 3-D lifting surfaces and wing-box structures",
    Aerospace Science and Technology 108 (2021) 106378.
"""

import math
from dataclasses import dataclass

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  NACA profile generator
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
#  AirfoilProfile dataclass
# ═══════════════════════════════════════════════════════════════════

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
        """Create from a NACA 4-digit code."""
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
            # Need to combine into: lower TE -> LE -> upper TE
            upper = coords[:len(coords) // 2]
            lower = coords[len(coords) // 2:]
            x = np.concatenate((lower[::-1, 0], upper[1:, 0]))
            z = np.concatenate((lower[::-1, 1], upper[1:, 1]))
        else:
            # Selig: upper TE -> LE -> lower TE (already contiguous)
            # Reverse to match our convention: lower TE -> LE -> upper TE
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

        # 3. Ensure topological closure for watertight volumes
        # Force the last point to be exactly the same as the first point
        px[-1], pz[-1] = px[0], pz[0]

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

        # Fit B-spline curve through points.
        # Degree 3–5, C2 continuity, 1e-3 mm tolerance.
        # This produces ~45–51 poles regardless of data point count, keeping
        # lofted surfaces compact for STEP export (~2–5 MB per wing).
        # WARNING: Do not increase max_degree above 5 or decrease tolerance
        # below 1e-3 — this causes B-spline pole explosion in lofted surfaces
        # (e.g. degree 8 + tol 1e-4 → 105 poles/wire → 157K poles/face → 122 MB).
        bspline = GeomAPI_PointsToBSpline(arr, 3, 5, GeomAbs_C2, 1e-3)
        if not bspline.IsDone():
            bspline = GeomAPI_PointsToBSpline(arr)

        edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()
        
        # Explicitly make wire and ensure it is recognized as closed
        mk_wire = BRepBuilderAPI_MakeWire(edge)
        if not mk_wire.IsDone():
             raise RuntimeError("Failed to create airfoil wire from B-spline edge")
             
        wire = mk_wire.Wire()
        return wire


# ═══════════════════════════════════════════════════════════════════
#  NurbsProfile dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NurbsProfile:
    """A highly controllable 2D NURBS airfoil profile.

    Attributes
    ----------
    poles : np.ndarray
        (N, 2) array of control points (x, z).
    knots : list or np.ndarray
        Knot vector (must be strictly increasing).
    multiplicities : list or np.ndarray
        Multiplicities for each knot.
    degree : int
        Degree of the B-spline basis functions.
    weights : list or np.ndarray or None
        Optional weights for a rational B-spline.
    name : str
        Descriptive name.
    chord : float
        Reference chord length these coordinates were generated at.
    num_eval_points : int
        Number of points to evaluate the curve at for the x and z arrays.
    """
    poles: np.ndarray
    knots: np.ndarray
    multiplicities: np.ndarray
    degree: int
    weights: np.ndarray = None
    name: str = ""
    chord: float = 1.0
    num_eval_points: int = 100

    def __post_init__(self):
        # Evaluate curve parametrically to fill x and z attributes
        # so this class can be used seamlessly in GVM
        self.poles = np.asarray(self.poles, dtype=float)
        self.knots = np.asarray(self.knots, dtype=float)
        self.multiplicities = np.asarray(self.multiplicities, dtype=int)
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)

        from OCP.Geom import Geom_BSplineCurve
        from OCP.gp import gp_Pnt
        from OCP.TColgp import TColgp_Array1OfPnt
        from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

        n_poles = len(self.poles)
        arr_poles = TColgp_Array1OfPnt(1, n_poles)
        for i, (x, z) in enumerate(self.poles):
            arr_poles.SetValue(i + 1, gp_Pnt(x, 0.0, z))

        n_knots = len(self.knots)
        arr_knots = TColStd_Array1OfReal(1, n_knots)
        for i, k in enumerate(self.knots):
            arr_knots.SetValue(i + 1, float(k))

        arr_mults = TColStd_Array1OfInteger(1, n_knots)
        for i, m in enumerate(self.multiplicities):
            arr_mults.SetValue(i + 1, int(m))

        if self.weights is not None:
            arr_weights = TColStd_Array1OfReal(1, n_poles)
            for i, w in enumerate(self.weights):
                arr_weights.SetValue(i + 1, float(w))
            curve = Geom_BSplineCurve(arr_poles, arr_weights, arr_knots, arr_mults, self.degree)
        else:
            curve = Geom_BSplineCurve(arr_poles, arr_knots, arr_mults, self.degree)

        u_min = curve.FirstParameter()
        u_max = curve.LastParameter()
        
        # you need a cosine clustering for evaluation to get denser points at LE/TE
        beta = np.linspace(0, math.pi, self.num_eval_points)
        u_norm = 0.5 * (1.0 - np.cos(beta))
        u_vals = u_min + u_norm * (u_max - u_min)

        x_list, z_list = [], []
        for u in u_vals:
            pt = curve.Value(u)
            x_list.append(pt.X())
            z_list.append(pt.Z())

        self.x = np.array(x_list)
        self.z = np.array(z_list)

    def scaled(self, new_chord):
        """Return a copy scaled to a different chord length."""
        factor = new_chord / self.chord
        weights_copy = self.weights.copy() if self.weights is not None else None
        return NurbsProfile(
            poles=self.poles * factor,
            knots=self.knots,
            multiplicities=self.multiplicities,
            degree=self.degree,
            weights=weights_copy,
            name=self.name,
            chord=new_chord,
            num_eval_points=self.num_eval_points
        )

    def to_occ_wire(self, position=(0.0, 0.0, 0.0), twist_deg=0.0, local_chord=None):
        from aeroshape.nurbs.utils import make_bspline_from_control_points
        
        if local_chord is not None and abs(local_chord - self.chord) > 1e-10:
            profile = self.scaled(local_chord)
        else:
            profile = self

        px, pz = profile.poles[:, 0].copy(), profile.poles[:, 1].copy()

        if abs(twist_deg) > 1e-10:
            le_idx = np.argmin(px)
            le_x, le_z = px[le_idx], pz[le_idx]
            angle = math.radians(twist_deg)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            dx = px - le_x
            dz = pz - le_z
            px = le_x + dx * cos_a + dz * sin_a
            pz = le_z - dx * sin_a + dz * cos_a

        x_off, y_pos, z_off = position
        
        # Ensure closure for watertight CAD volumes
        if len(px) > 2:
            px[-1], pz[-1] = px[0], pz[0]
            
        poles_3d = [(px[i] + x_off, y_pos, pz[i] + z_off) for i in range(len(px))]
        weights_list = list(self.weights) if self.weights is not None else None

        wire = make_bspline_from_control_points(
            poles_3d=poles_3d,
            knots=list(self.knots),
            multiplicities=list(self.multiplicities),
            degree=self.degree,
            weights=weights_list
        )
        return wire

