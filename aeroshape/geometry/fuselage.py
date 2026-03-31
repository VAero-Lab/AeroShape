"""Multi-segment fuselage definition and NURBS surface generation.

Provides:
- FuselageSegment: per-segment geometry parameters
- MultiSegmentFuselage: multi-segment fuselage builder

Coordinate convention:
    X = lengthwise (longitudinal), Y = spanwise (lateral), Z = thickness (vertical).
    Cross sections lie in the YZ plane and are lofted along the X axis.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np
import math

from aeroshape.geometry.cross_sections import CrossSectionProfile


def ellipsoid_blend(t):
    """Produces a perfectly rounded 1/4 ellipsoid lobe from t=0 to t=1."""
    return math.sqrt(1.0 - (1.0 - t)**2)

def inverse_paraboloid_blend(t):
    """Produces an upswept closing paraboloid boundary from t=0 (full) to t=1 (point)."""
    return 1.0 - math.sqrt(1.0 - t)

def _smooth_offset(t):
    """Cosine smooth ease-in/ease-out for offset transitions to guarantee tangent continuity."""
    return (1.0 - math.cos(t * math.pi)) / 2.0

@dataclass
class FuselageSegment:
    """Geometric specification for a single fuselage segment.

    Attributes
    ----------
    length : float
        Longitudinal length of this segment in meters.
    root_profile : CrossSectionProfile
        Cross-section profile at the segment root.
    tip_profile : CrossSectionProfile or None
        Cross-section profile at the segment tip. None means same as root.
    y_offset : float
        Lateral (spanwise) offset of the tip relative to the root.
    z_offset : float
        Vertical offset of the tip relative to the root (camber/sweep).
    num_sections : int
        Number of longitudinal stations within this segment.
    guide_curve_y : callable, optional
        A function f(t) -> float returning the nonlinear spanwise offset at parameter t in [0, 1].
    guide_curve_z : callable, optional
        A function f(t) -> float returning the nonlinear vertical offset at parameter t in [0, 1].
    """
    length: float
    root_profile: CrossSectionProfile
    tip_profile: Optional[CrossSectionProfile] = None
    y_offset: float = 0.0
    z_offset: float = 0.0
    num_sections: int = 10
    blend_curve: Optional[Callable[[float], float]] = None
    guide_curve_y: Optional[Callable[[float], float]] = None
    guide_curve_z: Optional[Callable[[float], float]] = None


@dataclass
class MultiSegmentFuselage:
    """Multi-segment fuselage builder.

    Chains FuselageSegment objects nose-to-tail and produces NURBS surfaces,
    structured vertex grids, and triangle meshes.

    Attributes
    ----------
    segments : list of FuselageSegment
        Ordered list of fuselage segments from nose to tail.
    name : str
        Descriptive name for the fuselage.
    """
    segments: List[FuselageSegment] = field(default_factory=list)
    name: str = "fuselage"

    def add_segment(self, segment: FuselageSegment) -> "MultiSegmentFuselage":
        """Append a segment and return self for chaining."""
        self.segments.append(segment)
        return self

    def get_section_frames(self, lengthwise_clustering=None) -> list:
        """Compute the 3D position and properties of every longitudinal station."""
        if hasattr(self, '_precomputed_frames') and self._precomputed_frames is not None:
            return self._precomputed_frames

        if not self.segments:
            return []

        frames = []
        x_accum = 0.0
        y_accum = 0.0
        z_accum = 0.0

        for seg_idx, seg in enumerate(self.segments):
            tip_profile = seg.tip_profile if seg.tip_profile is not None else seg.root_profile
            n = seg.num_sections
            start = 0 if seg_idx == 0 else 1

            if lengthwise_clustering is not None:
                t_values = lengthwise_clustering(n)
            else:
                t_values = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])

            for i in range(start, n):
                t = float(t_values[i])

                x = x_accum + t * seg.length
                
                # Apply explicit nonlinear guide curves if provided, else smooth transitioning (to prevent OCC loft protuberances)
                dy = seg.guide_curve_y(t) if seg.guide_curve_y else _smooth_offset(t) * seg.y_offset
                dz = seg.guide_curve_z(t) if seg.guide_curve_z else _smooth_offset(t) * seg.z_offset
                
                y = y_accum + dy
                z = z_accum + dz

                # Use nonlinear blend curve if provided for primitive shapes (ellipsoid/paraboloid)
                alpha = seg.blend_curve(t) if seg.blend_curve else t
                profile = _interpolate_profiles(seg.root_profile, tip_profile, alpha)

                frames.append({
                    "x": x,
                    "y_offset": y,
                    "z_offset": z,
                    "profile": profile,
                })

            # Update accumulation with the tip values
            x_accum += seg.length
            y_accum += seg.guide_curve_y(1.0) if seg.guide_curve_y else seg.y_offset
            z_accum += seg.guide_curve_z(1.0) if seg.guide_curve_z else seg.z_offset

        return frames

    def to_occ_shape(self):
        """Build a NURBS lofted surface through all section wires."""
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
        
        frames = self.get_section_frames()
        wires = []
        for fr in frames:
            wire = fr["profile"].to_occ_wire(
                position=(fr["x"], fr["y_offset"], fr["z_offset"])
            )
            wires.append(wire)

        # Shell (open ends), not a fully watertight solid by default.
        return NurbsSurfaceBuilder.loft(wires, solid=False, ruled=False)

    def to_vertex_grids(self, num_points_profile=50,
                        lengthwise_clustering=None,
                        profile_clustering=None):
        """Sample the NURBS lofted surface into structured (X, Y, Z) grids."""
        from aeroshape.nurbs.surfaces import NurbsSurfaceBuilder
        from aeroshape.nurbs.utils import sample_shape_grid

        frames = self.get_section_frames(lengthwise_clustering)
        if len(frames) < 2:
            raise ValueError("Need at least 2 section frames to loft")

        wires = []
        for fr in frames:
            wire = fr["profile"].to_occ_wire(
                position=(fr["x"], fr["y_offset"], fr["z_offset"])
            )
            wires.append(wire)

        # Fuselage is naturally a closed loft around the profile, but open at the ends (shell)
        shell = NurbsSurfaceBuilder.loft(wires, solid=False, ruled=False)

        n_lengthwise = len(frames)
        # Note the axis="X" here because the loft extends longitudinally along the X-axis
        X, Y, Z = sample_shape_grid(
            shell, n_lengthwise, num_points_profile,
            spanwise_clustering=lengthwise_clustering,
            chordwise_clustering=profile_clustering,
            axis='X'
        )
        return X, Y, Z

    def to_triangles(self, num_points_profile=50, closed=True,
                      lengthwise_clustering=None, profile_clustering=None):
        """Generate a triangle list from the vertex grids."""
        from aeroshape.analysis.mesh import MeshTopologyManager
        X, Y, Z = self.to_vertex_grids(num_points_profile, lengthwise_clustering, profile_clustering)
        return MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=closed)

    def compute_properties(self, method="gvm", density=1.0,
                           num_points_profile=50,
                           lengthwise_clustering=None,
                           profile_clustering=None):
        """Compute volume, mass, CG, and inertia using the chosen method."""
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
                                         lengthwise_clustering,
                                         profile_clustering)
        if method == "sai":
            volume = VolumeCalculator.compute_solid_volume_sai(X, Y, Z)
        else:
            from aeroshape.analysis.mesh import MeshTopologyManager
            triangles = MeshTopologyManager.get_wing_triangles(X, Y, Z, closed=True)
            volume = VolumeCalculator.compute_solid_volume(triangles)

        mass = volume * density
        cg, inertia, _ = MassPropertiesCalculator.compute_all(X, Y, Z, mass)

        return {
            "volume": volume,
            "mass": mass,
            "cg": cg,
            "inertia": inertia,
        }

def _interpolate_profiles(root: CrossSectionProfile, tip: CrossSectionProfile, t: float) -> CrossSectionProfile:
    """Linearly interpolate between two cross-section profiles."""
    ry = root.y
    rz = root.z
    ty = tip.y
    tz = tip.z

    # If point counts differ, resample tip to match root
    if len(ry) != len(ty):
        ty, tz = _resample_coords(ty, tz, len(ry))

    y = (1 - t) * ry + t * ty
    z = (1 - t) * rz + t * tz

    name = root.name if t < 0.5 else tip.name
    return CrossSectionProfile(y=y, z=z, name=name)

def _resample_coords(y: np.ndarray, z: np.ndarray, n: int):
    """Resample coordinate arrays to n points."""
    dy = np.diff(y)
    dz = np.diff(z)
    ds = np.sqrt(dy**2 + dz**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    s_norm = s / s[-1] if s[-1] > 0 else s
    s_new = np.linspace(0, 1, n)
    y_new = np.interp(s_new, s_norm, y)
    z_new = np.interp(s_new, s_norm, z)
    return y_new, z_new
