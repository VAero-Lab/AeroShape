"""Multi-segment wing definition and NURBS surface generation.

Provides:
- SegmentSpec: per-segment geometry parameters
- MultiSegmentWing: multi-segment wing builder

All produce OCC NURBS shapes, structured vertex grids (GVM pipeline),
and triangle meshes.

Coordinate convention:
    X = chordwise, Y = spanwise, Z = thickness.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from aeroshape.geometry.airfoils import AirfoilProfile


@dataclass
class SegmentSpec:
    """Geometric specification for a single wing segment.

    Attributes
    ----------
    span : float
        Spanwise length of this segment in meters.
    root_airfoil : AirfoilProfile
        Airfoil profile at the segment root.
    tip_airfoil : AirfoilProfile or None
        Airfoil profile at the segment tip. None means same as root.
    root_chord : float
        Chord length at the segment root in meters.
    tip_chord : float
        Chord length at the segment tip in meters.
    sweep_le_deg : float
        Leading-edge sweep angle in degrees.
    dihedral_deg : float
        Dihedral angle in degrees.
    twist_deg : float
        Tip twist relative to root (washout) in degrees.
    num_sections : int
        Number of spanwise stations within this segment.
    """
    span: float
    root_airfoil: AirfoilProfile
    tip_airfoil: Optional[AirfoilProfile] = None
    root_chord: float = 1.0
    tip_chord: float = 1.0
    sweep_le_deg: float = 0.0
    dihedral_deg: float = 0.0
    twist_deg: float = 0.0
    num_sections: int = 10


@dataclass
class MultiSegmentWing:
    """Multi-segment wing builder.

    Chains SegmentSpec objects root-to-tip and produces NURBS surfaces,
    structured vertex grids, and triangle meshes.

    Attributes
    ----------
    segments : list of SegmentSpec
        Ordered list of wing segments from root to tip.
    name : str
        Descriptive name for the wing.
    symmetric : bool
        If True, the wing is symmetric about Y=0 (only the starboard
        half is modeled).
    """
    segments: List[SegmentSpec] = field(default_factory=list)
    name: str = "wing"
    symmetric: bool = True

    def add_segment(self, segment: SegmentSpec) -> "MultiSegmentWing":
        """Append a segment and return self for chaining."""
        self.segments.append(segment)
        return self

    # ── Section frame computation ─────────────────────────────────

    def get_section_frames(self, spanwise_clustering=None) -> list:
        """Compute the 3D position and properties of every spanwise station.

        Walks the segment chain, accumulating sweep (x-offset), dihedral
        (z-offset), and interpolating twist, chord, and airfoil profile
        within each segment.

        The tip station of segment N equals the root station of segment N+1
        (shared boundary — no duplicate).

        If the wing was created via ``from_planform_curves``, the
        pre-computed frames are returned directly.

        Parameters
        ----------
        spanwise_clustering : callable or None
            A distribution function ``f(n) -> np.ndarray`` from
            :mod:`aeroshape.analysis.clustering` (e.g. ``cosine``,
            ``tanh_two_sided(2)``).  Applied within each segment to
            distribute sections non-uniformly along the span.  If *None*,
            uniform spacing is used.

        Returns
        -------
        frames : list of dict
            Each dict contains:
            - y : float — spanwise position
            - x_offset : float — chordwise offset from sweep
            - z_offset : float — vertical offset from dihedral
            - twist_deg : float — local twist angle
            - chord : float — local chord length
            - airfoil : AirfoilProfile — local airfoil profile
        """
        # Return pre-computed frames when built from guide curves
        if hasattr(self, '_precomputed_frames') and self._precomputed_frames is not None:
            return self._precomputed_frames

        if not self.segments:
            return []

        frames = []
        y_accum = 0.0
        x_accum = 0.0
        z_accum = 0.0
        twist_accum = 0.0

        for seg_idx, seg in enumerate(self.segments):
            tip_airfoil = seg.tip_airfoil if seg.tip_airfoil is not None else seg.root_airfoil

            n = seg.num_sections
            # First segment includes station 0; subsequent segments skip it
            # (shared with previous segment's tip).
            start = 0 if seg_idx == 0 else 1

            sweep_rad = math.radians(seg.sweep_le_deg)
            dihedral_rad = math.radians(seg.dihedral_deg)

            # Compute t-values (parameter within segment [0, 1])
            if spanwise_clustering is not None:
                t_values = spanwise_clustering(n)
            else:
                t_values = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])

            for i in range(start, n):
                t = float(t_values[i])

                y = y_accum + t * seg.span
                x_off = x_accum + t * seg.span * math.tan(sweep_rad)
                z_off = z_accum + t * seg.span * math.tan(dihedral_rad)
                twist = twist_accum + t * seg.twist_deg
                chord = (1 - t) * seg.root_chord + t * seg.tip_chord

                # Interpolate airfoil coordinates
                airfoil = _interpolate_airfoils(
                    seg.root_airfoil, tip_airfoil, t, chord
                )

                frames.append({
                    "y": y,
                    "x_offset": x_off,
                    "z_offset": z_off,
                    "twist_deg": twist,
                    "chord": chord,
                    "airfoil": airfoil,
                })

            # Accumulate for next segment
            y_accum += seg.span
            x_accum += seg.span * math.tan(sweep_rad)
            z_accum += seg.span * math.tan(dihedral_rad)
            twist_accum += seg.twist_deg

        return frames

    # ── NURBS shape ───────────────────────────────────────────────

    def to_occ_shape(self):
        """Build a NURBS lofted surface through all section wires.

        Returns
        -------
        TopoDS_Shape
            Lofted NURBS solid/shell from pythonocc.
        """
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
        return NurbsSurfaceBuilder.build(self)

    # ── Vertex grids (GVM pipeline) ──────────────────────────────

    def to_vertex_grids(self, num_points_profile=50,
                         spanwise_clustering=None,
                         chordwise_clustering=None):
        """Sample the NURBS lofted surface into structured (X, Y, Z) grids.

        Builds the NURBS loft through all section wires (same surface
        as ``to_occ_shape``), then evaluates it at parametric (u, v)
        coordinates.  This produces grids that follow the smooth C2
        NURBS surface rather than linearly interpolating between
        the defining section profiles.

        The number of spanwise stations is determined by the section
        frames, and clustering laws are applied in the parametric space
        of the lofted surface.

        Parameters
        ----------
        num_points_profile : int
            Number of chordwise points per profile (parameter m).
        spanwise_clustering : callable or None
            Distribution law ``f(n) -> array`` from
            :mod:`aeroshape.analysis.clustering`.
        chordwise_clustering : callable or None
            Distribution law ``f(n) -> array`` from
            :mod:`aeroshape.analysis.clustering`.

        Returns
        -------
        X, Y, Z : np.ndarray
            Coordinate matrices of shape (n_spanwise, num_points_profile).
        """
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
        from aeroshape.nurbs.utils import sample_shape_grid

        # Build the NURBS loft as a shell (no end caps) so you get
        # only the lateral surface for parametric sampling.
        frames = self.get_section_frames(spanwise_clustering)
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

        # Loft as shell (solid=False) to avoid end-cap faces
        shell = NurbsSurfaceBuilder.loft(wires, solid=False, ruled=False)

        n_spanwise = len(frames)
        X, Y, Z = sample_shape_grid(
            shell, n_spanwise, num_points_profile,
            spanwise_clustering=spanwise_clustering,
            chordwise_clustering=chordwise_clustering,
        )
        return X, Y, Z

    # ── Guide-curve construction ──────────────────────────────────

    @classmethod
    def from_planform_curves(cls, le_points, te_points, airfoil_stations,
                             num_sections=30, name="wing"):
        """Build a wing from leading- and trailing-edge guide curves.

        Instead of defining segments, the user specifies the leading-edge
        and trailing-edge shapes as ordered 3-D point lists.  Smooth
        B-spline curves are fitted through each set of points and sampled
        at ``num_sections`` stations.  At every station the chord is
        derived from the LE/TE distance and the airfoil shape is
        interpolated between the provided ``airfoil_stations``.

        This produces smooth planform transitions (no kinks at segment
        boundaries) and is especially useful for blended-wing-body
        configurations.

        Parameters
        ----------
        le_points : list of (float, float, float)
            3-D points defining the leading-edge curve, ordered root to
            tip.  Minimum 2 points.
        te_points : list of (float, float, float)
            3-D points defining the trailing-edge curve, ordered root to
            tip.  Minimum 2 points.
        airfoil_stations : list of (float, AirfoilProfile)
            Pairs of ``(spanwise_fraction, profile)`` where
            ``spanwise_fraction`` is in [0, 1] (0 = root, 1 = tip).
        num_sections : int
            Number of spanwise sections to create (more -> smoother).
        name : str
            Wing name.

        Returns
        -------
        MultiSegmentWing
            A wing whose ``get_section_frames()`` returns the computed
            frames (no ``SegmentSpec`` objects are needed).
        """
        le_curve = _fit_bspline_curve(le_points)
        te_curve = _fit_bspline_curve(te_points)

        le_u0, le_u1 = le_curve.FirstParameter(), le_curve.LastParameter()
        te_u0, te_u1 = te_curve.FirstParameter(), te_curve.LastParameter()

        stations_sorted = sorted(airfoil_stations, key=lambda s: s[0])

        frames = []
        for i in range(num_sections):
            t = i / (num_sections - 1) if num_sections > 1 else 0.0

            le_pt = le_curve.Value(le_u0 + t * (le_u1 - le_u0))
            te_pt = te_curve.Value(te_u0 + t * (te_u1 - te_u0))

            # Chord in XZ plane (exclude spanwise Y component)
            dx = te_pt.X() - le_pt.X()
            dz = te_pt.Z() - le_pt.Z()
            chord = math.sqrt(dx * dx + dz * dz)
            if chord < 1e-8:
                chord = 1e-8

            airfoil = _interpolate_airfoil_at_fraction(
                stations_sorted, t, chord
            )

            frames.append({
                "y": le_pt.Y(),
                "x_offset": le_pt.X(),
                "z_offset": le_pt.Z(),
                "twist_deg": 0.0,
                "chord": chord,
                "airfoil": airfoil,
            })

        wing = cls(name=name)
        wing._precomputed_frames = frames
        return wing

    # ── Triangle mesh ─────────────────────────────────────────────

    def to_triangles(self, num_points_profile=50, closed=True,
                      spanwise_clustering=None, chordwise_clustering=None):
        """Generate a triangle list from the vertex grids.

        Delegates to MeshTopologyManager.get_wing_triangles().

        Parameters
        ----------
        num_points_profile : int
            Number of chordwise points per profile.
        closed : bool
            If True, add end-cap triangles for a watertight mesh.
        spanwise_clustering : callable or None
            Distribution law for spanwise section spacing.
        chordwise_clustering : callable or None
            Distribution law for chordwise point spacing.

        Returns
        -------
        triangles : list of tuple
            Triangle list suitable for VolumeCalculator.
        """
        from aeroshape.analysis.mesh import MeshTopologyManager

        X, Y, Z = self.to_vertex_grids(num_points_profile,
                                         spanwise_clustering,
                                         chordwise_clustering)
        return MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=closed)

    # ── Property computation ────────────────────────────────────────

    def compute_properties(self, method="gvm", density=1.0,
                           num_points_profile=50,
                           spanwise_clustering=None,
                           chordwise_clustering=None):
        """Compute volume, mass, CG, and inertia using the chosen method.

        Parameters
        ----------
        method : str
            ``'gvm'`` — GVM Divergence-Theorem on triangulated mesh.
            ``'sai'`` — Section-Area Integration (shoelace + trapezoidal
            rule).  Eliminates chordwise inscribed-polygon error and is
            ~100 x faster than ``'gvm'``.
            ``'occ'`` — OCC BRepGProp on the exact NURBS surface.
        density : float
            Material density in kg/m^3.
        num_points_profile : int
            Chordwise resolution (used by GVM/SAI; ignored by OCC).
        spanwise_clustering : callable or None
            Spanwise distribution law (GVM/SAI only).
        chordwise_clustering : callable or None
            Chordwise distribution law (GVM/SAI only).

        Returns
        -------
        dict
            Keys: ``volume``, ``mass``, ``cg`` (3-array), ``inertia`` (6-tuple).
        """
        method = method.lower()
        if method == "occ":
            from aeroshape.nurbs.utils import occ_mass_properties
            shape = self.to_occ_shape()
            props = occ_mass_properties(shape, density)
            com = props["center_of_mass"]
            imat = props["inertia_matrix"]
            return {
                "volume": props["volume"],
                "mass": props["mass"],
                "cg": np.array(com),
                "inertia": (imat[0, 0], imat[1, 1], imat[2, 2],
                            imat[0, 1], imat[0, 2], imat[1, 2]),
            }

        from aeroshape.analysis.volume import VolumeCalculator
        from aeroshape.analysis.mass import MassPropertiesCalculator

        X, Y, Z = self.to_vertex_grids(num_points_profile,
                                         spanwise_clustering,
                                         chordwise_clustering)

        if method == "sai":
            volume = VolumeCalculator.compute_solid_volume_sai(X, Y, Z)
        else:
            # GVM Divergence-Theorem on NURBS-sampled geometry
            from aeroshape.analysis.mesh import MeshTopologyManager
            triangles = MeshTopologyManager.get_wing_triangles(
                X, Y, Z, closed=True)
            volume = VolumeCalculator.compute_solid_volume(triangles)

        mass = volume * density
        cg, inertia, _ = MassPropertiesCalculator.compute_all(
            X, Y, Z, mass)

        return {
            "volume": volume,
            "mass": mass,
            "cg": cg,
            "inertia": inertia,
        }


# ── Helpers ───────────────────────────────────────────────────────

def _interpolate_airfoils(root: AirfoilProfile, tip: AirfoilProfile,
                          t: float, chord: float) -> AirfoilProfile:
    """Linearly interpolate between two airfoil profiles.

    Both profiles must have the same number of points. The interpolation
    is done on unit-chord coordinates, then scaled to the target chord.

    Parameters
    ----------
    root, tip : AirfoilProfile
        The two profiles to blend.
    t : float
        Blend parameter (0 = root, 1 = tip).
    chord : float
        Target chord length for the result.
    """
    # Normalize both to unit chord for interpolation
    rx = root.x / root.chord if root.chord > 0 else root.x
    rz = root.z / root.chord if root.chord > 0 else root.z
    tx = tip.x / tip.chord if tip.chord > 0 else tip.x
    tz = tip.z / tip.chord if tip.chord > 0 else tip.z

    # If point counts differ, resample tip to match root
    if len(rx) != len(tx):
        n = len(rx)
        tx, tz = _resample_coords(tx, tz, n)

    x = ((1 - t) * rx + t * tx) * chord
    z = ((1 - t) * rz + t * tz) * chord

    name = root.name if t < 0.5 else tip.name
    return AirfoilProfile(x=x, z=z, name=name, chord=chord)


def _resample_profile(airfoil: AirfoilProfile, n: int,
                      chordwise_clustering=None) -> AirfoilProfile:
    """Resample an airfoil profile to n points preserving shape."""
    x_new, z_new = _resample_coords(airfoil.x, airfoil.z, n,
                                     clustering=chordwise_clustering)
    return AirfoilProfile(x=x_new, z=z_new, name=airfoil.name,
                          chord=airfoil.chord)


def _resample_coords(x: np.ndarray, z: np.ndarray, n: int, clustering=None):
    """Resample coordinate arrays to n points using arc-length interpolation.

    Parameters
    ----------
    x, z : np.ndarray
        Original coordinate arrays.
    n : int
        Target number of points.
    clustering : callable or None
        Distribution law ``f(n) -> array`` from :mod:`aeroshape.analysis.clustering`.
        If *None*, uniform spacing along arc length is used.
    """
    dx = np.diff(x)
    dz = np.diff(z)
    ds = np.sqrt(dx**2 + dz**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    s_norm = s / s[-1] if s[-1] > 0 else s

    s_new = clustering(n) if clustering is not None else np.linspace(0, 1, n)
    x_new = np.interp(s_new, s_norm, x)
    z_new = np.interp(s_new, s_norm, z)
    return x_new, z_new


def _fit_bspline_curve(points_3d):
    """Fit a smooth B-spline curve through a list of 3-D points.

    Uses OCC's ``GeomAPI_PointsToBSpline`` to produce a C2-continuous
    curve suitable for guide-curve wing construction.

    Parameters
    ----------
    points_3d : list of (float, float, float)
        Ordered 3-D points (at least 2).

    Returns
    -------
    Geom_BSplineCurve
        The fitted curve (OCC handle).
    """
    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.TColgp import TColgp_Array1OfPnt

    n = len(points_3d)
    arr = TColgp_Array1OfPnt(1, n)
    for i, pt in enumerate(points_3d):
        arr.SetValue(i + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

    bspline = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-4)
    if not bspline.IsDone():
        bspline = GeomAPI_PointsToBSpline(arr)
    return bspline.Curve()


def _interpolate_airfoil_at_fraction(stations, t, chord):
    """Interpolate an airfoil from a station list at spanwise fraction *t*.

    Parameters
    ----------
    stations : list of (float, AirfoilProfile)
        Sorted by spanwise fraction (0 = root, 1 = tip).
    t : float
        Spanwise fraction in [0, 1].
    chord : float
        Target chord length for the resulting profile.

    Returns
    -------
    AirfoilProfile
    """
    if not stations:
        raise ValueError("No airfoil stations defined")

    if len(stations) == 1 or t <= stations[0][0]:
        a = stations[0][1]
        return a.scaled(chord) if abs(a.chord - chord) > 1e-10 else a

    if t >= stations[-1][0]:
        a = stations[-1][1]
        return a.scaled(chord) if abs(a.chord - chord) > 1e-10 else a

    for j in range(len(stations) - 1):
        t0, a0 = stations[j]
        t1, a1 = stations[j + 1]
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0) if (t1 - t0) > 1e-10 else 0.0
            return _interpolate_airfoils(a0, a1, local_t, chord)

    a = stations[-1][1]
    return a.scaled(chord) if abs(a.chord - chord) > 1e-10 else a
