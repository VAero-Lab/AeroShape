"""Fuselage cross-section generation and representation.

Provides primitive shapes (circle, ellipse, arbitrary NURBS) for 
defining the lateral cross-sections of a fuselage.

Coordinate convention:
    For fuselages extending along the X-axis, the cross-sections
    lie in the YZ plane. By default, profiles are centered at Y=0, Z=0.
"""

import math
from dataclasses import dataclass
import numpy as np


from dataclasses import dataclass, field
import numpy as np


@dataclass
class CrossSectionProfile:
    """Base class for 2D fuselage cross sections.
    
    Attributes
    ----------
    y : np.ndarray
        Lateral (spanwise) coordinates of the cross section.
    z : np.ndarray
        Vertical (thickness) coordinates of the cross section.
    name : str
        Descriptive name.
    """
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    name: str = ""

    def to_occ_wire(self, position=(0.0, 0.0, 0.0)):
        """Convert to an OpenCASCADE B-spline wire.
        
        Places the profile in the YZ plane and translates it by `position`.
        
        Parameters
        ----------
        position : tuple
            (x_offset, y_offset, z_offset) in meters.
            
        Returns
        -------
        TopoDS_Wire
            Closed B-spline wire suitable for lofting.
        """
        from OCP.gp import gp_Pnt
        from OCP.GeomAPI import GeomAPI_PointsToBSpline
        from OCP.GeomAbs import GeomAbs_C2
        from OCP.TColgp import TColgp_Array1OfPnt
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

        py, pz = self.y.copy(), self.z.copy()
        x_off, y_off, z_off = position
        
        n = len(py)
        arr = TColgp_Array1OfPnt(1, n)
        for i in range(n):
            arr.SetValue(i + 1, gp_Pnt(
                float(x_off),
                float(py[i]) + y_off,
                float(pz[i]) + z_off,
            ))

        # Periodic (closed) curve is better for fuselages
        bspline = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-4)
        if not bspline.IsDone():
            bspline = GeomAPI_PointsToBSpline(arr)

        edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        return wire

    @classmethod
    def from_points(cls, y, z, name="custom"):
        """Create from explicit coordinate arrays."""
        return cls(
            y=np.asarray(y, dtype=float),
            z=np.asarray(z, dtype=float),
            name=name,
        )


class CircularProfile(CrossSectionProfile):
    """A circular fuselage cross-section."""
    
    def __init__(self, radius, num_points=60, name="Circle"):
        theta = np.linspace(0, 2 * math.pi, num_points)
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        super().__init__(y=y, z=z, name=name)


class EllipticalProfile(CrossSectionProfile):
    """An elliptical fuselage cross-section."""
    
    def __init__(self, width, height, num_points=60, name="Ellipse"):
        a = width / 2.0
        b = height / 2.0
        theta = np.linspace(0, 2 * math.pi, num_points)
        y = a * np.cos(theta)
        z = b * np.sin(theta)
        super().__init__(y=y, z=z, name=name)


from dataclasses import dataclass, field

@dataclass
class NurbsCrossSection(CrossSectionProfile):
    """A highly controllable 2D NURBS cross-section.
    
    Allows arbitrary closed or open shapes acting as cross-sections.
    """
    poles: np.ndarray = field(default_factory=lambda: np.array([]))
    knots: np.ndarray = field(default_factory=lambda: np.array([]))
    multiplicities: np.ndarray = field(default_factory=lambda: np.array([]))
    degree: int = 3
    weights: np.ndarray = None
    num_eval_points: int = 60

    def __post_init__(self):
        if len(self.poles) == 0:
            return
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
        for i, (y, z) in enumerate(self.poles):
            arr_poles.SetValue(i + 1, gp_Pnt(0.0, y, z))

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
        
        u_vals = np.linspace(u_min, u_max, self.num_eval_points)
        y_list, z_list = [], []
        for u in u_vals:
            pt = curve.Value(u)
            y_list.append(pt.Y())
            z_list.append(pt.Z())

        self.y = np.array(y_list)
        self.z = np.array(z_list)

    def to_occ_wire(self, position=(0.0, 0.0, 0.0)):
        from aeroshape.nurbs.utils import make_bspline_from_control_points
        py, pz = self.poles[:, 0].copy(), self.poles[:, 1].copy()
        x_off, y_off, z_off = position
        
        poles_3d = [(x_off, py[i] + y_off, pz[i] + z_off) for i in range(len(py))]
        weights_list = list(self.weights) if self.weights is not None else None

        wire = make_bspline_from_control_points(
            poles_3d=poles_3d,
            knots=list(self.knots),
            multiplicities=list(self.multiplicities),
            degree=self.degree,
            weights=weights_list,
            periodic=True if self.poles[0].tolist() == self.poles[-1].tolist() else False
        )
        return wire
