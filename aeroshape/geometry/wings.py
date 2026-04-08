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

            from OCP.gp import gp_Vec
            sweep_rad = math.radians(seg.sweep_le_deg)
            dihedral_rad = math.radians(seg.dihedral_deg)

            # Dynamic Frenet-Serret Basis for standard unguided segments
            v_span = gp_Vec(math.tan(sweep_rad), 1.0, math.tan(dihedral_rad)).Normalized()
            v_chord = gp_Vec(1, 0, 0)
            v_thickness = v_chord.Crossed(v_span)
            if v_thickness.Magnitude() > 1e-8:
                v_thickness.Normalize()
            else:
                v_thickness = gp_Vec(0, 0, 1)

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
                    "v_chord_dir": (v_chord.X(), v_chord.Y(), v_chord.Z()),
                    "v_thickness_dir": (v_thickness.X(), v_thickness.Y(), v_thickness.Z())
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
        Shape
            Lofted NURBS solid/shell (build123d object).
        """
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
        return NurbsSurfaceBuilder.build(self)

    def to_occ_segments(self, max_sections=15, spanwise_clustering=None):
        """Build individual NURBS lofts for cada segment, splitting large ones.

        Returns
        -------
        list of Shape
            List of lofted segments (build123d objects).
        """
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
        
        frames = self.get_section_frames(spanwise_clustering=spanwise_clustering)
        if len(frames) < 2:
            return []
            
        # Treat all frames as one continuous sequence and split into chunks
        # This enabled parallelization even for wings with one segment (like BWB)
        segments = []
        n_total = len(frames)
        
        # We walk through the frames and create lofts of max_sections overlapping by 1
        i = 0
        while i < n_total - 1:
            # End index for this chunk (at most max_sections away)
            j = min(i + max_sections, n_total)
            
            chunk_frames = frames[i:j]
            if len(chunk_frames) >= 2:
                wires = []
                for fr in chunk_frames:
                    wire = fr["airfoil"].to_occ_wire(
                        position=(fr["x_offset"], fr["y"], fr["z_offset"]),
                        twist_deg=fr["twist_deg"],
                        local_chord=fr["chord"],
                        v_chord_dir=fr.get("v_chord_dir"),
                        v_thickness_dir=fr.get("v_thickness_dir"),
                        two_edges=True
                    )
                    wires.append(wire)
                segments.append(NurbsSurfaceBuilder.loft(wires, solid=True))
            
            i = j - 1 # Next chunk starts at last frame of current chunk
            
        return segments

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
                v_chord_dir=fr.get("v_chord_dir"),
                v_thickness_dir=fr.get("v_thickness_dir")
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
    def from_planform_curves(cls, le_curve, te_curve, airfoil_stations,
                             num_sections=30, spanwise_clustering=None,
                             name="wing", symmetric=True):
        """Build a wing from leading- and trailing-edge guide curves.

        Instead of defining segments, the user specifies the leading-edge
        and trailing-edge shapes using `GuideCurve` objects from `aeroshape.geometry.curves`.
        These mathematically rigorous curves prevent approximation ringing artifacts.

        Parameters
        ----------
        le_curve : GuideCurve or list
            A `GuideCurve` builder object evaluating the wing leading edge,
            or a list of 3-D points (legacy behavior).
        te_curve : GuideCurve or list
            A `GuideCurve` builder object evaluating the wing trailing edge,
            or a list of 3-D points (legacy behavior).
        airfoil_stations : list of (float, AirfoilProfile)
            Pairs of ``(spanwise_fraction, profile)`` where
            ``spanwise_fraction`` is in [0, 1] (0 = root, 1 = tip).
        num_sections : int
            Number of spanwise sections to create (more -> smoother).
        name : str
            Wing name.
        symmetric : bool
            If True, only starboard half is provided and will be mirrored.
        spanwise_clustering : callable or None
            Distribution law for spanwise section spacing along the Guide Curves.
            Example: `aeroshape.analysis.clustering.tanh_two_sided()`.

        Returns
        -------
        MultiSegmentWing
            A wing whose ``get_section_frames()`` returns the computed
            frames (no ``SegmentSpec`` objects are needed).
        """
        from aeroshape.geometry.curves import GuideCurve
        
        if isinstance(le_curve, list):
            le_builder = GuideCurve(start_point=le_curve[0])
            le_builder.add_fitted_points(le_curve[1:])
            le_occ = le_builder.build_occ_curve()
        elif isinstance(le_curve, GuideCurve):
            le_occ = le_curve.build_occ_curve()
        else:
            le_occ = le_curve

        if isinstance(te_curve, list):
            te_builder = GuideCurve(start_point=te_curve[0])
            te_builder.add_fitted_points(te_curve[1:])
            te_occ = te_builder.build_occ_curve()
        elif isinstance(te_curve, GuideCurve):
            te_occ = te_curve.build_occ_curve()
        else:
            te_occ = te_curve

        le_curve = le_occ 
        te_curve = te_occ 

        le_u0, le_u1 = le_curve.FirstParameter(), le_curve.LastParameter()
        te_u0, te_u1 = te_curve.FirstParameter(), te_curve.LastParameter()

        stations_sorted = sorted(airfoil_stations, key=lambda s: s[0])

        from OCP.gp import gp_Pnt, gp_Vec
        import numpy as np
        
        if spanwise_clustering is not None:
            t_vals = spanwise_clustering(num_sections)
        else:
            t_vals = np.linspace(0.0, 1.0, num_sections) if num_sections > 1 else [0.0]

        frames = []
        for i in range(num_sections):
            t = float(t_vals[i])

            le_pt = gp_Pnt()
            le_vec = gp_Vec()
            le_curve.D1(le_u0 + t * (le_u1 - le_u0), le_pt, le_vec)

            te_pt = te_curve.Value(te_u0 + t * (te_u1 - te_u0))

            # 3D spatial chord length
            dx = te_pt.X() - le_pt.X()
            dy = te_pt.Y() - le_pt.Y()
            dz = te_pt.Z() - le_pt.Z()
            chord = math.sqrt(dx * dx + dy * dy + dz * dz)
            
            if chord < 1e-8:
                chord = 1e-8
                v_chord = gp_Vec(1, 0, 0)
            else:
                v_chord = gp_Vec(dx, dy, dz)

            # Frenet-Serret Moving Frame: 
            # Thickness is strictly orthogonal to both the Chord and the Spanwise geometric curve
            v_thickness = v_chord.Crossed(le_vec)
            if v_thickness.Magnitude() > 1e-8:
                v_thickness.Normalize()
            else:
                v_thickness = gp_Vec(0, 0, 1)

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
                "v_chord_dir": (v_chord.X(), v_chord.Y(), v_chord.Z()),
                "v_thickness_dir": (v_thickness.X(), v_thickness.Y(), v_thickness.Z()),
            })

        wing = cls(name=name, symmetric=symmetric)
        wing._precomputed_frames = frames
        return wing

    @classmethod
    def create_blended_winglet(cls, base_wing: "MultiSegmentWing", 
                               height_z: float, 
                               sweep_out_y: float = 0.5,
                               tip_chord_ratio: float = 0.5,
                               num_sections: int = 25,
                               spanwise_clustering=None,
                               name: str = "Blended Winglet"):
        """Create a G1 continuous winglet attaching precisely to the tip of an existing wing.
        
        Evaluates the terminal sweep and dihedral frame of the main wing and projects 
        a mathematically continuous quadratic Bezier loft upwards.
        """
        # Resolve the terminal boundary condition of the host wing
        frames = base_wing.get_section_frames()
        last = frames[-1]
        host_seg = base_wing.segments[-1] # The segment driving the outbound tangency
        
        le0 = (last["x_offset"], last["y"], last["z_offset"])
        chord0 = last["chord"]
        te0 = (le0[0] + chord0, le0[1], le0[2])
        
        # Terminal tangent extension dynamically bridging outward iteratively from host bounds
        d_out = sweep_out_y
        p1_le = (le0[0] + d_out * math.tan(math.radians(host_seg.sweep_le_deg)), 
                 le0[1] + d_out, 
                 le0[2] + d_out * math.tan(math.radians(host_seg.dihedral_deg)))
                 
        p1_te = (te0[0] + d_out * math.tan(math.radians(host_seg.sweep_le_deg)), 
                 te0[1] + d_out, 
                 te0[2] + d_out * math.tan(math.radians(host_seg.dihedral_deg)))

        # Winglet ascends directly along Z axis collinear to the tangent bridge point
        p2_le = (p1_le[0] + 0.5, p1_le[1], p1_le[2] + height_z)
        p2_te = (p1_te[0] + 0.5 - chord0*(1.0-tip_chord_ratio), p1_te[1], p1_te[2] + height_z)
        
        from aeroshape.geometry.curves import GuideCurve

        le = GuideCurve(start_point=le0)
        le.add_bezier([p1_le, p2_le])

        te = GuideCurve(start_point=te0)
        te.add_bezier([p1_te, p2_te])

        winglet = cls.from_planform_curves(
            le_curve=le, te_curve=te,
            airfoil_stations=[(0.0, last["airfoil"]), (1.0, last["airfoil"])],
            num_sections=num_sections, spanwise_clustering=spanwise_clustering, name=name
        )
        winglet.symmetric = base_wing.symmetric
        return winglet

    @classmethod
    def create_box_fin(cls, lower_wing: "MultiSegmentWing", upper_wing: "MultiSegmentWing", 
                       lower_origin: tuple = (0.0, 0.0, 0.0),
                       upper_origin: tuple = (0.0, 0.0, 0.0),
                       d_out: float = 3.0,
                       num_sections: int = 40,
                       spanwise_clustering=None,
                       name: str = "Box Wing Fin"):
        """Create a continuous flat-walled bounding box pillar securely marrying two wing topology boundaries.
        
        Extrapolates tangent vectors from both bounding conditions to lock analytic G1 evaluation, bypassing B-Spline ringing.
        Utilizes absolute bounding frames via lower_origin and upper_origin to correctly anchor the parametric loop.
        """
        lower_last = lower_wing.get_section_frames()[-1]
        upper_last = upper_wing.get_section_frames()[-1]
        
        le0 = (lower_last["x_offset"] + lower_origin[0], 
               lower_last["y"] + lower_origin[1], 
               lower_last["z_offset"] + lower_origin[2])
        te0 = (le0[0] + lower_last["chord"], le0[1], le0[2])
        
        le5 = (upper_last["x_offset"] + upper_origin[0], 
               upper_last["y"] + upper_origin[1], 
               upper_last["z_offset"] + upper_origin[2])
        te5 = (le5[0] + upper_last["chord"], le5[1], le5[2])
        
        flat_y = le0[1] + d_out
        
        lower_seg = lower_wing.segments[-1]
        p1_le = (le0[0] + d_out * math.tan(math.radians(lower_seg.sweep_le_deg)), flat_y, le0[2] + d_out * math.tan(math.radians(lower_seg.dihedral_deg)))
        p1_te = (te0[0] + d_out * math.tan(math.radians(lower_seg.sweep_le_deg)), flat_y, te0[2] + d_out * math.tan(math.radians(lower_seg.dihedral_deg)))
        
        upper_seg = upper_wing.segments[-1]
        # Invert the upper span bounds computationally to flow inbound effectively.
        p4_le = (le5[0] + d_out * math.tan(math.radians(-upper_seg.sweep_le_deg)), flat_y, le5[2] + d_out * math.tan(math.radians(-upper_seg.dihedral_deg)))
        p4_te = (te5[0] + d_out * math.tan(math.radians(-upper_seg.sweep_le_deg)), flat_y, te5[2] + d_out * math.tan(math.radians(-upper_seg.dihedral_deg)))
        
        p2_le = (p1_le[0] + 0.3*(p4_le[0]-p1_le[0]), flat_y, p1_le[2] + 0.3*(p4_le[2]-p1_le[2]))
        p3_le = (p1_le[0] + 0.7*(p4_le[0]-p1_le[0]), flat_y, p1_le[2] + 0.7*(p4_le[2]-p1_le[2]))
        
        p2_te = (p1_te[0] + 0.3*(p4_te[0]-p1_te[0]), flat_y, p1_te[2] + 0.3*(p4_te[2]-p1_te[2]))
        p3_te = (p1_te[0] + 0.7*(p4_te[0]-p1_te[0]), flat_y, p1_te[2] + 0.7*(p4_te[2]-p1_te[2]))
        
        from aeroshape.geometry.curves import GuideCurve

        le = GuideCurve(start_point=le0)
        le.add_bezier([p1_le, p2_le])
        le.add_line(end_point=p3_le)
        le.add_bezier([p4_le, le5])

        te = GuideCurve(start_point=te0)
        te.add_bezier([p1_te, p2_te])
        te.add_line(end_point=p3_te)
        te.add_bezier([p4_te, te5])

        fin = cls.from_planform_curves(
            le_curve=le, te_curve=te,
            airfoil_stations=[(0.0, lower_last["airfoil"]), (1.0, upper_last["airfoil"])],
            num_sections=num_sections, spanwise_clustering=spanwise_clustering, name=name
        )
        fin.symmetric = lower_wing.symmetric
        return fin

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
                           chordwise_clustering=None,
                           uproc=False, tolerance=None):
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
            if uproc:
                # Parallelize across segments using the same dispatcher as AircraftModel
                from aeroshape.nurbs.utils import occ_mass_properties
                from multiprocessing import Pool, cpu_count
                
                segments = self.to_occ_segments()
                with Pool(processes=cpu_count()) as pool:
                    results = pool.starmap(occ_mass_properties, [(s, density, tolerance) for s in segments])
                
                # Combine results
                total_volume = sum(r["volume"] for r in results)
                total_mass = sum(r["mass"] for r in results)
                
                if total_mass > 0:
                    cg_weighted_sum = np.zeros(3)
                    for r in results:
                        cg_weighted_sum += np.array(r["center_of_mass"]) * r["mass"]
                    total_cg = cg_weighted_sum / total_mass
                    
                    # Parallel Axis Theorem for inertia tensor summation
                    total_imat = np.zeros((3, 3))
                    for r in results:
                        mi = r["mass"]
                        ci = np.array(r["center_of_mass"])
                        # Inertia about global CG: I_global = I_local + m * ( |d|^2 * E - d * d^T )
                        d = ci - total_cg
                        d2 = np.dot(d, d)
                        steiner = mi * (d2 * np.eye(3) - np.outer(d, d))
                        total_imat += r["inertia_matrix"] + steiner
                else:
                    total_cg = np.zeros(3)
                    total_imat = np.zeros((3, 3))

                return {
                    "volume": total_volume,
                    "mass": total_mass,
                    "cg": total_cg,
                    "inertia": (total_imat[0, 0], total_imat[1, 1], total_imat[2, 2],
                                total_imat[0, 1], total_imat[0, 2], total_imat[1, 2]),
                }
            else:
                shape = self.to_occ_shape()
                from aeroshape.nurbs.utils import occ_mass_properties
                props = occ_mass_properties(shape, density, tolerance)
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

    def show(self, method='occ', uproc=True, tolerance=0.1, props=None, **kwargs):
        """Launch the high-fidelity interactive CAD viewer for this wing.

        Parameters
        ----------
        method : str
            Analysis method (ignored if `props` is provided).
        uproc : bool
            Enable parallel analysis across wing segments.
        tolerance : float
            Integration tolerance.
        props : dict or None
            Manual properties dictionary (volume, mass, cg, inertia).
        **kwargs : dict
            Additional arguments for `show_interactive`.
        """
        from aeroshape.visualization.rendering import show_interactive
        if props is None:
            props = self.compute_properties(method=method, uproc=uproc, tolerance=tolerance)
        
        # 2. Get NURBS sampling grids (standard lattice look)
        X, Y, Z = self.to_vertex_grids(num_points_profile=80)
        grids = [(X, Y, Z, self.name, False)]
        
        # 3. Launch viewer
        show_interactive(
            grids, 
            props['volume'], props['mass'], props['cg'], props['inertia'],
            title=kwargs.pop('title', self.name),
            **kwargs
        )


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
